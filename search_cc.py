import cdx_toolkit

cdx = cdx_toolkit.CDXFetcher(source='cc')
queries = ['example.com', 'example.org']  # Add more queries to this list.

for query in queries:
    print("query:", query)
    obj_list = list(cdx.iter(url=query, from_ts='202001', to='202101', limit=5))
    print('\n{} search results for {}:'.format(len(obj_list), query))
    for obj in obj_list:
        print(obj.data)