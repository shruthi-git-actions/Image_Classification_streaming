from xml.dom.minidom import parse
import dvc.api

xmldata = dvc.api.read('get-started/data.xml',
    repo='https://github.com/iterative/dataset-registry')
xmldom = parse(xmldata)