#gsutil ls -lh gs://active-ird-experiments/rss-logs/logs/output | sort -k 3
#filedir="200107"
#filedir=test.txt
filedir="200110"
#ls "$abc"
gsutil -m cp -r "gs://active-ird-experiments/rss-logs/logs/output/$filedir/*" "data/$filedir/"
