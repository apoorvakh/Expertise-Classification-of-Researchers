#-------------------------------------------------------------------------------
# Name:        module2
# Purpose:
#
# Author:      Apoorva
#
# Created:     08/06/2016
# Copyright:   (c) Apoorva 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------


import urllib
url = 'http://export.arxiv.org/api/query?search_query=cat:cs.AI'
data = urllib.urlopen(url).read()
print data