import unittest

from gsplat_trainer.eval.eval_dataloader import EvalDataLoader


class EvalDataLoaderTest(unittest.TestCase):
    def test_given_a_nonexistent_data_dir_and_dataset__when_getting_the_split__then_throw_an_error(
        self,
    ) -> None:
        with self.assertRaises(Exception):
            EvalDataLoader("data", "test").get_eval_split()
