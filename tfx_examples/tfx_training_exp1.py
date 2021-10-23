from tfx.components import StatisticsGen
from tfx.orchestration.experimental.interactive.interactive_context import \
    InteractiveContext

context = InteractiveContext()
statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
context.run(statistics_gen)
context.show(statistics_gen.outputs['statistics'])
for artifact in statistics_gen.outputs['statistics'].get():
    print(artifact.uri)
