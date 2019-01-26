import time
import huffman as H
# files =["short.txt"]
files =["book.txt","dan.bmp", "music.wav"]
for f in files:
    start = time.time()
    H.compress(f, f + ".huf")
    print("compressed {} in {} seconds.".format(f, time.time() - start))
    start = time.time()
    fc = f + ".huf"
    H.uncompress(fc, fc + ".orig")
    print("uncompressed {} in {} seconds.".format(fc, time.time() - start))
