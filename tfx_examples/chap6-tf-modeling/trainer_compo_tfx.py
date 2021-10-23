from tfx.components import Trainer
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.proto import trainer_pb2
import os

TRAINING_STEP = 1000
EVAL_STEP= 100
trainer = Trainer(
    module_file=os.path.abspath("tf_model_exp.py"),
    custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
    #transformed_examples= transform.outputs['transformed_examples']
)