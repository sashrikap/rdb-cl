#gsutil ls -lh gs://active-ird-experiments/rss-logs/logs/output | sort -k 3
#filedir="200107"
filedir=test.txt
#ls "$abc"
gsutil cp -r "gs://active-ird-experiments/rss-logs/logs/output/$filedir" data/
