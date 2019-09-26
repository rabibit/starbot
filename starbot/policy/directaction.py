import logging
from typing import Optional, Any, List, Text
from rasa.core.domain import Domain
from rasa.core.featurizers import TrackerFeaturizer
from rasa.core.policies.policy import Policy
from rasa.core.trackers import DialogueStateTracker
from rasa.core.events import UserUttered, SlotSet, BotUttered
import os

from starbot.policy.gpt2_extractor.gpt2_extractor import Gpt2Extractor
from starbot.utils.colors import reset, Fg

logger = logging.getLogger(__name__)


class DirectAction(Policy):
    """The policy that invoke action according to intent directly
    """
    gpt2_extractor: Gpt2Extractor

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
        message: UserUttered = tracker.events[-1]
        if tracker.events and isinstance(message, UserUttered):
            # 如果bot最后说的话里面有提示备选项，用gpt2辅助提取ner
            index = domain.index_for_action('action_process_intent')
            result[index] = 1.0
            prompt = self.get_bot_prompt(tracker)

            logger.info(f'user uttered:{Fg.red}{message.text}{reset}')
            logger.info(f'prompt is {Fg.red}{prompt}{reset}')

            if prompt:
                gpt2out = self.gpt2_extractor.process(prompt, message.text)
                logger.info(f'gpt2out:{Fg.red}{gpt2out}{reset}')
                message.parse_data['gpt2out'] = gpt2out.splitlines(keepends=False)[0]
        return result

    @staticmethod
    def get_bot_prompt(tracker: DialogueStateTracker):
        count = 0
        for event in reversed(tracker.events):
            if isinstance(event, BotUttered):
                if event.text.startswith('/'):
                    continue
                prompt = event.metadata.get('prompt')
                if prompt:
                    return prompt
                count += 1
                if count >= 3:
                    return None

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


