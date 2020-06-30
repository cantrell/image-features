from multiprocessing import Process, Queue
from os import path
import os
from Runner import Runner
import json

OUTPUT_DIR = 'output'
CHUNK_SIZE = 100
INPUT_FILE = './500k_With_Meta+Annotations.json'

def main():
    if (path.exists(OUTPUT_DIR) == False):
        os.mkdir(OUTPUT_DIR)

    outputFileCount = len([name for name in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, name))])
    start = (outputFileCount * CHUNK_SIZE)

    with open(INPUT_FILE, 'r', encoding='utf-8') as jsonFile:
        data = json.load(jsonFile)
        total = len(data)

    iterations = int((total / CHUNK_SIZE) - outputFileCount)

    for i in range(0, iterations):
        q = Queue()
        p = Process(target=loop, args=(q, (i * CHUNK_SIZE) + start, data, ))
        p.start()
        featureList = q.get()
        p.join()

        low = ((i * CHUNK_SIZE) + start) + 1
        high = (low + CHUNK_SIZE) - 1

        outputFileName = '%d-%d.json' % (low, high)
        with open(os.path.join(OUTPUT_DIR, outputFileName), 'w') as outfile:
            print('Writing to output file: %s' % outputFileName)
            json.dump(featureList, outfile)

    print('Done!')

def loop(q, start, data):
    runner = Runner(start, (start + CHUNK_SIZE) - 1, data)
    featureList = runner.run()
    q.put(featureList)

if __name__ == "__main__":
    main()