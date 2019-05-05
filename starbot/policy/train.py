#!/usr/bin/env python


if __name__ == "__main__":
    from rasa_core.train import train

    train(domain_file="domain.yml",
          stories_file="stories.md",
          output_path="models",
          policy_config="policy_config.yml",
          kwargs={
              "debug_plots": True
          })
