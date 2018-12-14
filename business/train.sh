#!/bin/sh

python -m rasa_core.train -o models -d domain.yml -s stories.md

