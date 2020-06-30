from ImageFeatures import ImageFeatures
from PIL import Image, ImageOps, UnidentifiedImageError
import requests
from io import BytesIO
import json

class Runner:
    def __init__(self, start, end):
        print('Creating new Runner')
        self.start = start
        self.end = end
        self.imgF = ImageFeatures()

    def run(self):
        print('Runner.run()')
        with open('./500k_With_Meta+Annotations.json', 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        print('Starting from %d.' % (self.start + 1))
        featureList = {}
        for i in range(self.start, self.end + 1):
            item = data[i]
            itemURL = item['medium_url']
            itemID = item['cid']
            print('Processing %d of %d (%s)' % (i, self.end, itemURL))
            try:
                response = requests.get(itemURL)
            except requests.HTTPError as e:
                status_code = e.response.status_code
                print(status_code)
                exit(-1)
                # TODO Don't need to exit; we could just try again
            try:
                image = Image.open(BytesIO(response.content))
            except UnidentifiedImageError as e:
                print('Cannot open this image:')
                print(e)
                continue
            ratio = float(item['h_1000']) / float(item['w_1000'])
            scale = float(item['w_1000']) / image.size[0]
            coords = item['annotations']['annotation'][0]['coordinates']
            angle = float(item['annotations']['annotation'][0]['angle'])
            print(ratio)
            print(coords)
            print(image.size)
            centerX = (coords['x']) / float(item['w_1000'])
            centerY = (coords['y']) / float(item['h_1000'])
            print(centerX, centerY)
            flatImage = ImageOps.fit( image, ( int(coords['w'] / scale),int(coords['h'] / scale) ), method=Image.NEAREST, bleed=0, centering=(centerX, centerY) )

            try:
                features = self.imgF.getFeatures([flatImage])
            except ValueError as e:
                print('ImageFeatures.getFeatures failed...')
                print(e)
                print('We\'ll just skip it.')
                continue

            featureList[itemID] = features.tolist()
            print('----------')
        return featureList