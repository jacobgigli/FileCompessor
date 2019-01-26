"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode



# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    freq = {}
    for b in text:
        freq[b] = freq.get(b, 0)+1
    return freq

def first_coordinate(x): # helper function used to sort on first element
    return x[0]

def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    """

    node_list = [] # list of tuples : (freq,HuffmanNode)
    for symbol in freq_dict:
        node_list.append((freq_dict[symbol], HuffmanNode(symbol)))

    node_list.sort(key=first_coordinate)

    n = len(node_list) - 1
    for _ in range(n):
        left = node_list.pop(0)
        right = node_list.pop(0)
        node_list.append((left[0] + right[0],\
                          HuffmanNode(None, left[1], right[1])))
        node_list.sort(key=first_coordinate)

    assert len(node_list) == 1
    return node_list.pop(0)[1]

def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    code_dict = {}
    prefix = ""

    def generate_codes(tree, prefix, code_dict):
        if tree.is_leaf():
            code_dict[tree.symbol] = prefix
        else:
            generate_codes(tree.left, prefix + "0", code_dict)
            generate_codes(tree.right, prefix + "1", code_dict)

    generate_codes(tree, prefix, code_dict)
    return code_dict


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    def po_num(tree, current):
        if tree.is_leaf():
            return current
        else:
            tree.number = po_num(tree.right, po_num(tree.left, current))
        return tree.number+1 # the new current number

    po_num(tree, 0)


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    code_dict = get_codes(tree)

    total_bytes = 0
    total_compressed = 0
    for symbol in freq_dict:
        total_bytes += freq_dict[symbol]
        total_compressed += len(code_dict[symbol])*freq_dict[symbol]

    return total_compressed/total_bytes


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    compressed_text_bytes = bytes(b"")
    compressed_text = ""

    for b in text:
        compressed_text = compressed_text + codes[b]
        if len(compressed_text) % 8 == 0:  # at byte boundary so write out bytes
            while len(compressed_text) > 0:
                compressed_text_bytes = compressed_text_bytes + \
                                    bytes([bits_to_byte(compressed_text[0:8])])
                compressed_text = compressed_text[8:]

    while len(compressed_text) > 8:
        compressed_text_bytes = compressed_text_bytes + \
                                bytes([bits_to_byte(compressed_text[0:8])])
        compressed_text = compressed_text[8:]

    compressed_text_bytes = compressed_text_bytes + \
                            bytes([bits_to_byte(compressed_text)])

    return compressed_text_bytes



def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    def root_to_bytes(root):
        l_type = 0 if root.left.is_leaf() else 1
        l_symbol = root.left.symbol if root.left.is_leaf() else root.left.number
        r_type = 0 if root.right.is_leaf() else 1
        r_symbol = root.right.symbol if root.right.is_leaf() \
                                     else root.right.number
        return bytes([l_type, l_symbol, r_type, r_symbol])

    byte_list = bytes(b"")
    if tree.left and not tree.left.is_leaf():
        byte_list = byte_list + tree_to_bytes(tree.left)
    if tree.right and not tree.right.is_leaf():
        byte_list = byte_list + tree_to_bytes(tree.right)
    byte_list = byte_list + root_to_bytes(tree)

    return byte_list

def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2) == \
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None),\
    HuffmanNode(12, None, None)), HuffmanNode(None, HuffmanNode(5, None, None),\
    HuffmanNode(7, None,None)))
    True
    """
    def generate_subtree(interior, data, node_lst):
        return HuffmanNode(data) if interior == 0 \
            else generate_tree_general(node_lst, data)

    root = node_lst[root_index]
    ltree = generate_subtree(root.l_type, root.l_data, node_lst)
    rtree = generate_subtree(root.r_type, root.r_data, node_lst)
    return HuffmanNode(None, ltree, rtree)




def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2) == \
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None),\
    HuffmanNode(7, None, None)),\
    HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    True
    """

    root = node_lst[root_index]

    if root.r_type == 0 and root.l_type == 0:
        return HuffmanNode(None, HuffmanNode(root.l_data),\
                           HuffmanNode(root.r_data))
    else:
        if root.r_type == 0:
            return HuffmanNode(None, \
                               generate_tree_postorder(node_lst, root_index-1),\
                               HuffmanNode(root.r_data))
        elif root.l_type == 0:
            return HuffmanNode(None, HuffmanNode(root.l_data), \
                   generate_tree_postorder(node_lst,
                                           root_index - 1))
        else:
            rtree = generate_tree_postorder(node_lst, root_index - 1)
            number_nodes(rtree)
            offset = rtree.number+2
            ltree = generate_tree_postorder(node_lst, root_index -  offset)

            return HuffmanNode(None, ltree, rtree)


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes
    """

    restored_text = b""
    current = tree
    text_index = 0
    bit_text = ""

    for i in range(len(text)):
        bit_text += byte_to_bits(text[i])

    for _ in range(size):
        while not current.is_leaf():
            fork = bit_text[text_index]
            text_index += 1
            current = current.left if fork == "0" else current.right


        assert current.is_leaf()
        restored_text += bytes([current.symbol])
        current = tree

    return restored_text

def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        # tree = generate_tree_postorder(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """

    def generate_depth_list(tree, freq_dict, depth_list, depth):
        if tree.is_leaf():
            depth_list.append([freq_dict[tree.symbol], depth, tree])
        if tree.left:
            generate_depth_list(tree.left, freq_dict, depth_list, depth+1)
        if tree.right:
            generate_depth_list(tree.right, freq_dict, depth_list, depth+1)

    depth_list = []
    generate_depth_list(tree, freq_dict, depth_list, 0)
    depth_list.sort(reverse=True, key=first_coordinate)

    while len(depth_list) > 1:
        min_depth = depth_list[0][1]
        min_depth_index = 0
        for j in range(1, len(depth_list)):
            if depth_list[j][1] < min_depth:
                min_depth = depth_list[j][1]
                min_depth_index = j

        depth_list[0][2].symbol, depth_list[min_depth_index][2].symbol = \
            depth_list[min_depth_index][2].symbol, depth_list[0][2].symbol

        depth_list[0][0], depth_list[min_depth_index][0] = \
            depth_list[min_depth_index][0], depth_list[0][0]

        del depth_list[min_depth_index]
        depth_list.sort(reverse=True, key=first_coordinate)


if __name__ == "__main__":



    import time

    import sys


    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
