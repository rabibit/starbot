#!/usr/bin/env python


if __name__ == "__main__":
    from rasa_core.train import train_dialogue_model
    train_dialogue_model(domain_file="domain.yml",
                         stories_file="stories.md",
                         output_path="models",
                         policy_config="policy_config.yml")
