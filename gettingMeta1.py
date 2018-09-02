#!/usr/bin/env python

# get the arxiv id
import sys
from string import strip, split
for arg in sys.argv[1:]:
    arg = strip(arg)
    arg = strip(arg, chars="arxiv:")
    arg = strip(arg, chars="http://")
    arg = strip(arg, chars="www.")
    arg = strip(arg, chars="arxiv.org/abs/")
    arg = split(arg, sep='v')[0]
    xid = strip(arg)

    # download the xml
    import urllib
    from xml.dom import minidom
    usock = urllib.urlopen('http://export.arxiv.org/api/query?id_list='+xid)
    xmldoc = minidom.parse(usock)
    usock.close()

    print xmldoc.toxml()
    print ""

    d = xmldoc.getElementsByTagName("entry")[0]

    date = d.getElementsByTagName("updated")[0].firstChild.data
    text_year = date[:4]

    title = d.getElementsByTagName("title")[0]
    text_title = title.firstChild.data#.encode('ascii', 'ignore')

    authorlist = []
    first = True
    for person_name in d.getElementsByTagName("author"):
        # get names
        name = person_name.getElementsByTagName("name")[0]
        text_name = name.firstChild.data#.encode('ascii', 'ignore')
        text_given_name = ' '.join(text_name.split()[:-1])
        text_surname = text_name.split()[-1]
        authorlist.append(text_surname+", "+text_given_name)
        #first author?
        if first:
            text_first_author_surname = text_surname
            first = False

    # output

    print "@MISC{"+text_first_author_surname+text_year[-2:]+","
    print "author = {"+" and ".join(authorlist)+"},"
    print "title = {"+text_title+"},"
    print "year = {"+text_year+"},"
    print "eprint = {"+xid+"},"
    print "URL = {http://www.arxiv.org/abs/"+xid+"},"
    print "}"