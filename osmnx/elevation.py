################################################################################
# Module: elevation.py
# Description: Get node elevations and edge grades from the Google Maps
#              Elevation API
# License: MIT, see full license in LICENSE.txt
# Web: https://github.com/gboeing/osmnx
################################################################################

import math
import time
import requests
import pandas as pd
import numpy as np
import networkx as nx

from .core import save_to_cache
from .core import get_from_cache
from .utils import log


def fetch_elevations_for_latitude_longitude_pairs(latitude_longitude_pairs,
                                                  url_template_kwargs,
                                                  decimal_places_to_round=5,
                                                  max_locations_per_batch=350,
                                                  pause_duration=0.02,
                                                  url_template='https://maps.googleapis.com/maps/api/elevation/json?locations={locations}&key={api_key}',
                                                  proxies=dict(),
                                                  ):
    """
    Get the elevation (meters) of each point in the DataFrame and add it to the
    row as an attribute.

    Parameters
    ----------
    latitude_longitude_pairs : pd.DataFrame
        the points to fetch elevation for;
        this DataFrame must have columns 'latitude' and 'longitude'
    url_template_kwargs : dict
        **url_template_kwargs will be passed to url_template.format.
        The latitude-longitude pairs will be stringified and added
        under the key 'locations'.
        Usually, this will just contain your google maps elevation API key
        under the key 'api_key'.
    decimal_places_to_round : int
        round coorindates to some number of decimal places
        to fit more locations in each API call
    max_locations_per_batch : int
        max number of coordinate pairs to submit in each API call (if this is
        too high, the server will reject the request because its character
        limit exceeds the max)
    pause_duration : float
        time to pause between API calls
    url_template : string
        URL to query for elevation data.
        Unless you have an unusual network setup, let this default to
        the standard Google Maps elevation API endpoint.
    proxies : dict
        proxies parameter for requests.get;
        see `the requests documentation <https://2.python-requests.org//en/master/user/advanced/#proxies>`_.

    Returns
    -------
    latitude_longitude_pairs : pd.DataFrame
    """
    log('Requesting node elevations from the API in {} calls.'.format(math.ceil(len(latitude_longitude_pairs) / max_locations_per_batch)))
    # break the series of coordinates into chunks of size max_locations_per_batch
    # API format is locations=lat,lng|lat,lng|lat,lng|lat,lng...
    latlng_pair_format_string = '{:.' + str(decimal_places_to_round) + 'f},{:.' + str(decimal_places_to_round) + 'f}'
    results = []
    for i in range(0, len(latitude_longitude_pairs), max_locations_per_batch):
        chunk = latitude_longitude_pairs.iloc[i : i + max_locations_per_batch]
        locations = '|'.join(latlng_pair_format_string.format(row.latitude, row.longitude) for row in chunk.itertuples())
        url = url_template.format(locations, **url_template_kwargs)

        # check if this request is already in the cache (if global use_cache=True)
        cached_response_json = get_from_cache(url)
        if cached_response_json is not None:
            response_json = cached_response_json
        else:
            try:
                # request the elevations from the API
                log('Requesting node elevations: {}'.format(url))
                time.sleep(pause_duration)
                response = requests.get(url, proxies=proxies)
                response_json = response.json()
                save_to_cache(url, response_json)
            except Exception as e:
                log(e)
                log('Server responded with {}: {}'.format(response.status_code, response.reason))

        # append these elevation results to the list of all results
        results.extend(response_json['results'])

    # sanity check that all our vectors have the same number of elements
    if not (len(results) == len(latitude_longitude_pairs)):
        raise Exception('We requested {} elevations but we received {} results from the elevation API.'.format(len(latitude_longitude_pairs), len(results)))
    else:
        log('We requested {} elevations and we received {} results from the elevation API.'.format(len(latitude_longitude_pairs), len(results)))

    # add elevation as an attribute to the DataFrame
    latitude_longitude_pairs['elevation'] = [result['elevation'] for result in results]
    return latitude_longitude_pairs

def add_node_elevations(G, api_key,
                        max_locations_per_batch=350,
                        pause_duration=0.02,
                        url_template='https://maps.googleapis.com/maps/api/elevation/json?locations={}&key={}',
                        proxies=dict(),
                        ): # pragma: no cover
    """
    Get the elevation (meters) of each node in the network and add it to the
    node as an attribute.

    Parameters
    ----------
    G : networkx multidigraph
    api_key : string
        your google maps elevation API key
    max_locations_per_batch : int
        max number of coordinate pairs to submit in each API call (if this is
        too high, the server will reject the request because its character
        limit exceeds the max)
    pause_duration : float
        time to pause between API calls
    url_template : string
        URL to query for elevation data.
        Unless you have an unusual network setup, let this default to
        the standard Google Maps elevation API endpoint.
    proxies : dict
        proxies parameter for requests.get;
        see `the requests documentation <https://2.python-requests.org//en/master/user/advanced/#proxies>`_.

    Returns
    -------
    G : networkx multidigraph
    """

    node_latitude_longitude_triples = ((node, data['y'], data['x']) for (node, data) in G.nodes(data=True))
    df = pd.DataFrame(node_latitude_longitude_triples, columns=('node_id', 'latitude', 'longitude'))
    assert df.node_id.dtype == np.int64
    assert df.latitude.dtype == np.float64
    assert df.longitude.dtype == np.float64
    df.set_index('node_id', inplace=True)
    # round coorindates to 5 decimal places (approx 1 meter) to be able to fit
    # in more locations per API call
    df = fetch_elevations_for_latitude_longitude_pairs(df, dict(api_key=api_key),
                                                       decimal_places_to_round=5,
                                                       max_locations_per_batch=max_locations_per_batch,
                                                       pause_duration=pause_duration,
                                                       url_template=url_template,
                                                       proxies=proxies,
                                                       )

    # sanity check that all our vectors have the same number of elements
    if not (len(df['elevation']) == len(G.nodes())):
        raise Exception('Graph has {} nodes but we received {} results from the elevation API.'.format(len(G.nodes()), len(df['elevation'])))
    else:
        log('Graph has {} nodes and we received {} results from the elevation API.'.format(len(G.nodes()), len(df['elevation'])))

    df['elevation'] = df['elevation'].round(3) # round to millimeter
    # add elevation as an attribute to the nodes
    nx.set_node_attributes(G, name='elevation', values=df['elevation'].to_dict())
    log('Added elevation data to all nodes.')

    return G



def add_edge_grades(G, add_absolute=True): # pragma: no cover
    """
    Get the directed grade (ie, rise over run) for each edge in the network and
    add it to the edge as an attribute. Nodes must have elevation attributes to
    use this function.

    Parameters
    ----------
    G : networkx multidigraph
    add_absolute : bool
        if True, also add the absolute value of the grade as an edge attribute

    Returns
    -------
    G : networkx multidigraph
    """

    # for each edge, calculate the difference in elevation from origin to
    # destination, then divide by edge length
    for u, v, data in G.edges(keys=False, data=True):
        elevation_change = G.nodes[v]['elevation'] - G.nodes[u]['elevation']
        
        # round to ten-thousandths decimal place
        grade = round(elevation_change / data['length'], 4)
        data['grade'] = grade
        if add_absolute:
            data['grade_abs'] = abs(grade)

    log('Added grade data to all edges.')
    return G
