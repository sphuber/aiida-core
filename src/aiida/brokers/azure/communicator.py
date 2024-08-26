import asyncio
import collections
import concurrent.futures
import functools
import json
import weakref
from concurrent.futures import Future as ThreadFuture
from contextlib import asynccontextmanager

import kiwipy
from azure.servicebus import ServiceBusMessage
from azure.servicebus.aio import ServiceBusClient
from kiwipy import Communicator, Future
from pytray import aiothreads  # type: ignore[import-untyped]

from aiida.common.log import AIIDA_LOGGER

LOGGER = AIIDA_LOGGER.getChild('broker.azure')


class AzureThreadCommunicator(Communicator):
    """."""

    @classmethod
    def connect(cls, connection_string: str):
        return cls(connection_string)

    def __init__(self, connection_string: str):
        self._loop = asyncio.new_event_loop()
        self._loop.set_debug(True)

        self._loop_scheduler = aiothreads.LoopScheduler(self._loop, 'AzureServiceBus communicator')
        self._loop_scheduler.start()  # Start the loop scheduler (i.e. the event loop thread)

        self._connection_string = connection_string
        self._communicator = AzureCommunicator(connection_string=connection_string)

    def __str__(self):
        return str(self._communicator)

    def task_send(self, task, no_reply=False):
        return self._loop_scheduler.await_(self._communicator.task_send(task, no_reply))

    def add_task_subscriber(self, subscriber, identifier=None):
        print('AzureThreadCommunicator.add_task_subscriber', subscriber)
        wrapped_subscriber = self._wrap_subscriber(subscriber)
        return self._loop_scheduler.await_submit(self._communicator.add_task_subscriber(wrapped_subscriber, identifier))

    def _wrap_subscriber(self, subscriber):
        """ "
        We need to convert any kiwipy.Futures we get from a subscriber call into asyncio ones for
        the event loop based communicator.  Do this by wrapping any subscriber methods and
        intercepting the return values.
        """

        @functools.wraps(subscriber)
        def wrapper(*args, **kwargs):
            result = subscriber(*args, **kwargs)
            if isinstance(result, ThreadFuture):
                result = self._wrap_future(result)
            return result

        return wrapper

    def _wrap_future(self, kiwi_future: Future):
        aio_future = self._loop.create_future()

        def done(_):
            try:
                result = kiwi_future.result()
            except concurrent.futures.CancelledError:
                self._loop.call_soon_threadsafe(aio_future.cancel)
            except Exception as exc:  # pylint: disable=broad-except
                self._loop.call_soon_threadsafe(aio_future.set_exception, exc)
            else:
                if isinstance(result, Future):
                    result = self._wrap_future(result)
                self._loop.call_soon_threadsafe(aio_future.set_result, result)

        kiwi_future.add_done_callback(done)
        return aio_future


class AzureCommunicator(Communicator):
    def __init__(self, connection_string: str):
        self.task_queue_name = 'test-sph'
        self._connection_string = connection_string
        self._client = ServiceBusClient.from_connection_string(conn_str=self._connection_string, logging_enable=True)

    def __str__(self):
        return f'AzureCommunicator<{self._connection_string}>'

    @asynccontextmanager
    async def get_queue_sender(self, queue_name: str):
        async with self._client:
            async with self._client.get_queue_sender(queue_name=queue_name) as sender:
                yield sender

    async def task_send(self, task, no_reply=False):
        print(f'{self.__class__.__name__}.task_send', task)
        async with self.get_queue_sender(self.task_queue_name) as sender:
            await sender.send_messages(ServiceBusMessage(json.dumps(task)))

    async def repeat(self, interval, func, *args, **kwargs):
        """Run func every interval seconds.

        If func has not finished before *interval*, will run again
        immediately when the previous iteration finished.

        *args and **kwargs are passed as the arguments to func.
        """
        while True:
            await asyncio.gather(func(*args, **kwargs), asyncio.sleep(interval))

    async def add_task_subscriber(self, subscriber, identifier=None):
        await asyncio.ensure_future(self.repeat(10, functools.partial(self.check_for_messages, subscriber)))

    async def check_for_messages(self, subscriber):
        print('CHECKING FOR MESSAGES')
        async with self._client:
            async with self._client.get_queue_receiver(queue_name=self.task_queue_name) as receiver:
                messages = await receiver.receive_messages(max_wait_time=5)
                for message in messages:
                    try:
                        await self.on_message(receiver, subscriber, message)
                    except Exception as exception:
                        print('HANDLING MESSAGE EXCEPTED', exception)
                        await receiver.abandon_message(message)

    async def on_message(self, receiver, subscriber, message):
        from kiwipy.rmq import utils

        decoded = json.loads(str(message.raw_amqp_message))
        print('ON MESSAGE', decoded, subscriber)

        try:
            task = AzureIncomingTask(receiver, subscriber, message)
        except Exception as e:
            print('EXCEPTION', e)
            raise
        print('ON MESSAGE TASK', task)
        async with task.processing() as outcome:
            print('PROCESSING MESSAGE')
            try:
                subscriber = utils.ensure_coroutine(subscriber)
                print('CALLING SUBSCRIBER')
                result = await subscriber(self, decoded)
                print('SUBSCRIBER RESULT', result)

                # If a task returns a future it is not considered done until the chain of
                # futures (i.e. if the first future resolves to a future and so on) finishes
                # and produces a concrete result
                while asyncio.isfuture(result):
                    print('RESULT', type(result))
                    if not task.no_reply:
                        print('SENDING RESPONSE')
                        await self._send_response(utils.pending_response(), message)
                        print('SENDING RESPONSE DONE')
                    result = await result
            except kiwipy.TaskRejected:
                # Task was rejected by this subscriber, keep trying
                pass
            except kiwipy.CancelledError:
                # The subscriber has cancelled their processing of the task
                outcome.cancel()
            except Exception as exc:  # pylint: disable=broad-except
                # There was an exception during the processing of this task
                outcome.set_exception(exc)
                LOGGER.exception('Exception occurred while processing task.')
            else:
                # All good
                outcome.set_result(result)

            print('MESSAGE', decoded, subscriber)
            return await subscriber(self, decoded)


TaskInfo = collections.namedtuple('TaskInfo', ('task', 'no_reply'))

TASK_PENDING = 'pending'
TASK_FINISHED = 'finished'
TASK_PROCESSING = 'processing'
TASK_REQUEUED = 'requeued'


class AzureIncomingTask:
    def __init__(self, receiver, subscriber, message):
        self._receiver = receiver
        self._subscriber = subscriber
        self._message = message
        self._task_info = TaskInfo(*json.loads(str(message.raw_amqp_message)))
        self._state = TASK_PENDING
        self._outcome_ref: weakref.ReferenceType | None = None
        self._loop = asyncio.get_event_loop()

    @property
    def body(self) -> str:
        return self._task_info.task

    @property
    def no_reply(self) -> bool:
        return self._task_info.no_reply

    @property
    def state(self) -> str:
        return self._state

    def process(self) -> asyncio.Future:
        if self._state != TASK_PENDING:
            raise asyncio.InvalidStateError(f'The task is {self._state}')

        self._state = TASK_PROCESSING
        outcome = self._loop.create_future()
        # Rely on the done callback to signal the end of processing
        outcome.add_done_callback(self._on_task_done)
        # Or the user lets the future get destroyed
        self._outcome_ref = weakref.ref(outcome, self._outcome_destroyed)

        return outcome

    async def requeue(self):
        if self._state not in [TASK_PENDING, TASK_PROCESSING]:
            raise asyncio.InvalidStateError(f'The task is {self._state}')

        self._state = TASK_REQUEUED
        await self._receiver.abandon_message(self._message)
        self._finalise()

    @asynccontextmanager
    async def processing(self):
        # async def processing(self) -> t.Generator[asyncio.Future, None, None]:
        """Processing context.  The task should be done at the end otherwise it's assumed the
        caller doesn't want to process it, and it's sent back to the queue"""

        print('PROCESSING', self._state)
        if self._state != TASK_PENDING:
            raise asyncio.InvalidStateError(f'The task is {self._state}')

        self._state = TASK_PROCESSING
        print('PROCESSING', self._state)
        outcome = self._loop.create_future()
        try:
            print('PROCESSING YIELD')
            yield outcome
        except KeyboardInterrupt:  # pylint: disable=try-except-raise
            raise
        except Exception as exc:
            # Set the exception on the task and re-raise so the client also sees it
            outcome.set_exception(exc)
            raise
        finally:
            print('PROCESSING FINALLY')
            if outcome.done():
                await self._task_done(outcome)
            else:
                await self.requeue()

    def _on_task_done(self, outcome):
        """Schedule a task to call ``_task_done`` when the outcome is done."""
        self._loop.create_task(self._task_done(outcome))

    async def _task_done(self, outcome: asyncio.Future):
        assert outcome.done()
        self._outcome_ref = None

        if outcome.cancelled():
            # Whoever took the task decided not to process it
            self._state = TASK_PENDING
        else:
            # Task is done or excepted
            # Permanently store the outcome
            self._state = TASK_FINISHED
            await self._receiver.complete_message(self._message)

            # # We have to get the result from the future here (even if not replying), otherwise
            # # python complains that it was never retrieved in case of exception
            # try:
            #     reply_body = utils.result_response(outcome.result())
            # except Exception:  # pylint: disable=broad-except
            #     reply_body = utils.exception_response(exc)

            # if not self.no_reply:
            #     # Schedule a task to send the appropriate response
            #     # pylint: disable=protected-access
            #     await self._subscriber._send_response(reply_body, self._message)

        # Clean up
        self._finalise()

    def _outcome_destroyed(self, outcome_ref):
        # This only happens if someone called self.process() and then let the future
        # get destroyed without setting an outcome
        assert outcome_ref is self._outcome_ref
        # This task will not be processed
        self._outcome_ref = None
        asyncio.run_coroutine_threadsafe(self.requeue(), loop=self._loop)

    def _finalise(self):
        self._outcome_ref = None
        self._subscriber = None
        self._message = None
