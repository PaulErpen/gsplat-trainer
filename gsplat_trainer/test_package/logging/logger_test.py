import unittest

from gsplat_trainer.logging.logger_factory import LoggerFactory


class LoggerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = LoggerFactory.create_mocked_logger()

    def test_given_a_mocked_logger__when_calling_the_logger_once__then_exactly_one_call_must_be_registered(
        self,
    ) -> None:
        self.logger.log({}, 0)

        self.assertEqual(len(self.logger.logger_backend.calls), 1)
    
    def test_given_a_mocked_logger__when_calling_the_logger_twice__then_exactly_two_calls_must_be_registered(
        self,
    ) -> None:
        self.logger.log({}, 0)
        self.logger.log({}, 0)

        self.assertEqual(len(self.logger.logger_backend.calls), 2)
