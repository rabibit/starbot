import logging
from typing import Optional, Any, List, Text
from rasa.core.domain import Domain
from rasa.core.featurizers import TrackerFeaturizer
from rasa.core.policies.policy import Policy
from rasa.core.trackers import DialogueStateTracker
from rasa.core.events import UserUttered, SlotSet
import os

from starbot.policy.gpt2_extractor.gpt2_extractor import Gpt2Extractor

logger = logging.getLogger(__name__)


class DirectAction(Policy):
    """The policy that invoke action according to intent directly
    """

    def __init__(
            self,
            featurizer: Optional[TrackerFeaturizer] = None,
            priority: int = 4,
    ) -> None:
        super(DirectAction, self).__init__(featurizer, priority)

    def train(
            self,
            training_trackers: List[DialogueStateTracker],
            domain: Domain,
            **kwargs: Any
    ) -> None:
        """Trains the policy on given training trackers."""
        pass

    def continue_training(
            self,
            training_trackers: List[DialogueStateTracker],
            domain: Domain,
            **kwargs: Any
    ) -> None:
        pass

    def predict_action_probabilities(
            self, tracker: DialogueStateTracker, domain: Domain
    ) -> List[float]:
        """Predicts the next action the bot should take
            after seeing the tracker.

            Returns the list of probabilities for the next actions.
            If memorized action was found returns 1.1 for its index,
            else returns 0.0 for all actions."""
        result = [0.0] * domain.num_actions
        message:UserUttered = tracker.events[-1]
        if tracker.events and isinstance(message, UserUttered):
            index = domain.index_for_action('action_process_intent')
            result[index] = 1.0
            prompt = tracker.get_slot('gpt2prompt')
            if prompt:
                gpt2out = self.gpt2_extractor.process(prompt, message.text)
                message.parse_data['gpt2out'] = gpt2out
                tracker.update(SlotSet('gpt2prompt', None))
        return result

    def persist(self, path: Text) -> None:
        logger.debug(f"path is {path}")
        os.system(f"mkdir -p '{path}'")
        os.system(f"cp -r ../gpt2/* '{path}'")
        return None

    @classmethod
    def load(cls, path: Text) -> "DirectAction":
        slf = cls()
        slf.gpt2_extractor = Gpt2Extractor.load(path)
        return slf


