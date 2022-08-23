from icrawler.builtin import GoogleImageCrawler
import argparse

app = argparse.ArgumentParser()
app.add_argument("--key_word","-k",type=str)
app.add_argument("--path","-p",type=str)
app.add_argument("--max_num","-mn",type=int,help="--key_word 'keyword' --path 'path_to_store' --max_num '(Int) max to download'")

args = app.parse_args()

google_crawler = GoogleImageCrawler(downloader_threads=1000,storage={'root_dir':args.path})
google_crawler.crawl(keyword=args.key_word,max_num=args.max_num)
