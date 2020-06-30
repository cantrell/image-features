from multiprocessing import Process, Queue
from os import path
import os
from Runner import Runner
import json

OUTPUT_DIR = 'output'
CHUNK_SIZE = 100
TOTAL = 131293

def main():
    if (path.exists(OUTPUT_DIR) == False):
        os.mkdir(OUTPUT_DIR)

    outputFileCount = len([name for name in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, name))])
    start = (outputFileCount * CHUNK_SIZE)

    iterations = int((TOTAL / CHUNK_SIZE) - outputFileCount)

    for i in range(0, iterations):
        q = Queue()
        p = Process(target=loop, args=(q, (i * CHUNK_SIZE) + start, ))
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

def loop(q, start):
    runner = Runner(start, (start + CHUNK_SIZE) - 1)
    featureList = runner.run()
    q.put(featureList)

if __name__ == "__main__":
    main()