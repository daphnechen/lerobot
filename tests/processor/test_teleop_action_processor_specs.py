"""A teleop action pipeline has to be reachable from a config file.

Before this, the pipeline was an optional Python argument to
build_rollout_context, so any teleoperator needing a non-identity step also
needed its own launcher script duplicating lerobot-rollout. Worse, a plain
`lerobot-rollout --config_path=...` did not fail -- it silently substituted
IdentityProcessorStep and passed raw device deltas through as if they were
absolute poses.
"""

import dataclasses

import draccus
import pytest

from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import make_teleop_action_processor_from_specs
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.rollout.configs import ProcessorStepSpec


@pytest.fixture
def scaling_step():
    """A registered step with a tunable field, unregistered afterwards.

    Deliberately not the real SpaceMouse step: this behaviour belongs to the
    registry mechanism, and a test that imported a robot plugin to exercise it
    would fail for reasons that have nothing to do with the mechanism.
    """

    @ProcessorStepRegistry.register(name="_test_scale_action")
    @dataclasses.dataclass
    class ScaleAction(ProcessorStep):
        gain: float = 1.0

        def __call__(self, transition: EnvTransition) -> EnvTransition:
            action = transition.get(TransitionKey.ACTION.value)
            if action is None:
                return transition
            out = dict(transition)
            out[TransitionKey.ACTION.value] = {k: v * self.gain for k, v in action.items()}
            return out

        def get_config(self) -> dict:
            return {"gain": self.gain}

        def transform_features(self, features):
            return features

    yield ScaleAction
    ProcessorStepRegistry.unregister("_test_scale_action")


def test_no_specs_yields_the_identity_pipeline():
    """Callers pass the config field through unconditionally, so empty must work."""
    pipeline = make_teleop_action_processor_from_specs([])
    assert [type(s).__name__ for s in pipeline.steps] == ["IdentityProcessorStep"]


def test_a_named_step_is_built_and_actually_runs(scaling_step):
    pipeline = make_teleop_action_processor_from_specs(
        [ProcessorStepSpec(name="_test_scale_action", kwargs={"gain": 2.0})]
    )
    assert isinstance(pipeline.steps[0], scaling_step)

    # Built is not enough: the step has to be reached by the pipeline's
    # transition conversion, which is where a wrong to_transition would show.
    assert pipeline(({"j": 1.5}, {"j": 0.0})) == {"j": 3.0}


def test_kwargs_are_optional_and_leave_step_defaults_alone(scaling_step):
    pipeline = make_teleop_action_processor_from_specs(
        [ProcessorStepSpec(name="_test_scale_action")]
    )
    assert pipeline.steps[0].gain == 1.0


def test_steps_compose_in_the_order_given(scaling_step):
    pipeline = make_teleop_action_processor_from_specs([
        ProcessorStepSpec(name="_test_scale_action", kwargs={"gain": 2.0}),
        ProcessorStepSpec(name="_test_scale_action", kwargs={"gain": 3.0}),
    ])
    assert len(pipeline.steps) == 2
    assert pipeline(({"j": 1.0}, {"j": 0.0})) == {"j": 6.0}


def test_an_unknown_name_says_a_plugin_may_be_unimported():
    """The likeliest cause of a miss, and the one the registry cannot know."""
    with pytest.raises(KeyError, match="imported"):
        make_teleop_action_processor_from_specs([ProcessorStepSpec(name="_nonexistent")])


def test_draccus_decodes_the_yaml_shape(scaling_step):
    """The field is only useful if it survives the config parser."""
    specs = draccus.decode(
        list[ProcessorStepSpec],
        [{"name": "_test_scale_action", "kwargs": {"gain": 4.0}}],
    )
    assert specs == [ProcessorStepSpec(name="_test_scale_action", kwargs={"gain": 4.0})]

    pipeline = make_teleop_action_processor_from_specs(specs)
    assert pipeline(({"j": 1.0}, {"j": 0.0})) == {"j": 4.0}
