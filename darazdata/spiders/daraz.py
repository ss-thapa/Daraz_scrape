import scrapy
from pathlib import Path

class DarazSpider(scrapy.Spider):
    name = "daraz"
    allowed_domains = ["daraz.com.np"]
    start_urls = ["https://daraz.com.np"]


    def start_requests(self):
        urls = [
            "https://www.daraz.com.np/products/gravity-airshot-bassbuds-pro-40-hrs-playback-enc-bluetooth-53-quad-mic-i128431802-s1035922183.html?spm=a2a0e.searchlist.sku.6.57c9441b3Um3T5&search=1"
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url
        filename = f"quotes-{page}.html"
        Path(filename).write_bytes(response.body)
        self.log(f"Saved file {filename}")
