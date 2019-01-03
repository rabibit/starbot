ROOT=`pwd`
cd nlu
python -m rasa_core.run -vv -d $ROOT/business/models -u $ROOT/nlu/models/current/nlu --endpoints $ROOT/business/endpoints.yml --port 5002 --credentials $ROOT/credentials.yml
