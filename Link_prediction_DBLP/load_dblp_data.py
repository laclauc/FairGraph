import json
import os
import pickle
import random
from os.path import join
from collections import defaultdict
import numpy as np

from tqdm import tqdm

import pycountry
import pycountry_convert


def load_venues(data_name):
    if data_name == 'ml_dm':
        return [
            {'raw': 'knowledge discovery and data mining', 'id': '1130985203', 'acronym': 'KDD'},
            {'raw': 'european conference on principles of data mining and knowledge discovery', 'id': '1141769385',
             'acronym': 'ECML-PKDD'},
            {'raw': 'Data Mining and Knowledge Discovery', 'id': '121920818', 'acronym': 'DAMI'},
            {'raw': 'neural information processing systems', 'id': '1127325140', 'acronym': 'NIPS'},
            {'raw': 'Journal of Machine Learning Research', 'id': '118988714', 'acronym': 'JMLR'},
            {'raw': 'international conference on machine learning', 'id': '1180662882', 'acronym': 'ICML'},
            {'raw': 'Machine Learning', 'id': '62148650', 'acronym': 'MLJ'},
            {'raw': 'international conference on learning representations', 'id': '2584161585', 'acronym': 'ICLR'}]
    else:
        raise ValueError("Wrong data name.")


def from_cache(cache_file):
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        return None


def to_cache(cache_file, data):
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_paper_records(data_folder, cache_folder, target_venues):
    paper_records_file = join(cache_folder, 'paper_records.pkl')
    paper_records = from_cache(paper_records_file)
    if paper_records is None:
        target_vids = [venue['id'] for venue in target_venues]
        data_file = join(data_folder, "dblp_papers_v11.txt")
        paper_records = defaultdict(dict)
        with open(data_file) as f:
            for i, line in tqdm(enumerate(f)):
                prec = json.loads(line)
                try:
                    if prec['venue']['id'] in target_vids:
                        paper_records[prec['id']] = prec
                except KeyError:
                    pass
        to_cache(paper_records_file, paper_records)
    return paper_records


def load_author_records(paper_records):
    author_records = defaultdict(dict)
    for pid, prec in paper_records.items():
        for author in prec['authors']:
            aid = author['id']
            if aid not in author_records:
                author_records[aid]['name'] = set()
                author_records[aid]['orgs'] = set()
                author_records[aid]['num_papers'] = 0
            author_records[aid]['num_papers'] += 1
            author_records[aid]['name'].add(author['name'])
            if 'org' in author:
                author_records[aid]['orgs'].add(author['org'])
    return author_records


def subsample_paper_records(paper_records, subsample_step):
    result = {}
    count = 0
    for pid, prec in paper_records.items():
        if count % subsample_step == 0:
            result[pid] = prec
        count += 1
    return result


def load_dblp_data(data_name, data_folder, cache_folder, subsample_step):
    if not os.path.isdir(cache_folder):
        os.makedirs(cache_folder)

    dblp_data_file = join(cache_folder, "dblp_data.pkl")
    dblp_data = from_cache(dblp_data_file)
    if dblp_data is None:
        target_venues = load_venues(data_name)
        paper_records = load_paper_records(data_folder, cache_folder, target_venues)
        paper_records = subsample_paper_records(paper_records, subsample_step)

        author_records = load_author_records(paper_records)

        aid_id_dict = {aid: i for i, aid in enumerate(author_records.keys())}
        id_aid_dict = {i: aid for i, aid in enumerate(author_records.keys())}

        pid_id_dict = {pid: i for i, pid in enumerate(paper_records.keys())}
        id_pid_dict = {i: pid for i, pid in enumerate(paper_records.keys())}

        dblp_data = {
            'data_name': data_name,
            'paper_records': paper_records,
            'author_records': author_records,
            'target_venues': target_venues,
            'aid_id_dict': aid_id_dict,
            'id_aid_dict': id_aid_dict,
            'pid_id_dict': pid_id_dict,
            'id_pid_dict': id_pid_dict
        }
        to_cache(dblp_data_file, dblp_data)
    return dblp_data


def compute_edgelist(dblp_data, network_type, cache_folder):
    network_compilors = {
        'author-author': compile_ca_network
    }
    edgelist_file = join(cache_folder, '{:s}.edgelist'.format(network_type))
    if not os.path.exists(edgelist_file):
        E = network_compilors[network_type](dblp_data)
        np.savetxt(edgelist_file, E, fmt='%d', delimiter=' ')


def compile_ca_network(dblp_data):
    paper_records = dblp_data['paper_records']
    author_records = dblp_data['author_records']
    aid_id_dict = dblp_data['aid_id_dict']
    E = []

    times_author_found = 0
    times_author_not_found = 0
    for pid, prec in paper_records.items():
        ids = []
        for author in prec['authors']:
            aid = author['id']
            if aid in author_records:
                ids.append(aid_id_dict[aid])
                times_author_found += 1
            else:
                times_author_not_found += 1
        for i in range(len(ids) - 1):
            for j in range(i + 1, len(ids)):
                E.append([ids[i], ids[j]])
                # E.append([ids[j], ids[i]])
    E = np.array(E)
    print(times_author_found)
    print(times_author_not_found)
    return E


def get_countries_and_states():
    country_dict = {}  # a dict map each country name to a most common name
    state_abbv_list = ['AK', 'AL', 'AR', 'AS', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE',
                       'FL', 'GA', 'GU', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD',
                       'ME', 'MI', 'MN', 'MO', 'MP', 'MS', 'MT', 'NA', 'NC', 'ND', 'NE', 'NH', 'NJ',
                       'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX',
                       'UT', 'VA', 'VI', 'VT', 'WA', 'WI', 'WV', 'WY']

    for state in state_abbv_list:
        country_dict[state] = state

    for country in list(pycountry.countries):
        country_dict[country.name] = country.name
        if hasattr(country, 'official_name'):
            country_dict[country.official_name] = country.name

        if hasattr(country, 'common_name'):
            country_dict[country.common_name] = country.name
            country_dict[country.official_name] = country.name
            country_dict[country.name] = country.name

    country_dict['US'] = 'United States'
    country_dict['USA'] = 'United States'
    country_dict['America'] = 'United States'
    country_dict['UK'] = 'United Kingdom'

    for state in state_abbv_list:
        country_dict[state] = 'United States'

    return country_dict


def get_author_with_country(author_records):
    country_dict = get_countries_and_states()
    country_list = list(country_dict.keys())
    filtered_authors = {}
    aid_attr_dict = {}

    for aid, author in author_records.items():
        if 'orgs' not in author or len(author['orgs']) == 0:
            continue

        # For all listed organizations, extract the continents of the countries that are mentioned.
        canonical_continents = set()
        for org in author['orgs']:
            org = org.replace('#TAB#', '')

            tokenized_org_list = [e.strip(", ") for e in org.replace('#TAB#', '').split()]
            countries = list(set(tokenized_org_list) & set(country_list))
            for country in countries:
                canonical_country = country_dict[country]
                # Try to disambiguate 'Georgia'.
                if canonical_country == "Georgia":
                    if "Georgia Institute of Technology".lower() in org.lower() or \
                            "Georgia Tech".lower() in org.lower() or \
                            "Atlanta".lower() in org.lower():
                        canonical_country = "United States"
                    if "Tbilisi".lower() in org.lower():
                        canonical_country = "Georgia"

                # Convert country to continent.
                alpha2_code = pycountry_convert.country_name_to_country_alpha2(canonical_country)
                continent_code = pycountry_convert.country_alpha2_to_continent_code(alpha2_code)
                continent = pycountry_convert.convert_continent_code_to_continent_name(continent_code)
                canonical_continents.add(continent)

        # Check that exactly one continent is found for an author.
        if len(canonical_continents) != 1:
            continue

        # Get that continent.
        continent = next(iter(canonical_continents))
        aid_attr_dict[aid] = continent

        # If the author survived all checks, it is added to the 'filtered' authors.
        filtered_authors[aid] = author

    all_used_continents, continent_counts = np.unique(list(aid_attr_dict.values()), return_counts=True)
    print(all_used_continents)
    print(continent_counts)

    infrequent_countries = all_used_continents[continent_counts < 10]
    aid_to_remove = set()
    for aid, country in aid_attr_dict.items():
        if country in infrequent_countries:
            aid_to_remove.add(aid)

    print("Out of {} authors for which a country was found, {} were removed."
          .format(len(filtered_authors), len(aid_to_remove)))
    for aid in aid_to_remove:
        aid_attr_dict.pop(aid)
        filtered_authors.pop(aid)

    # Cast country attribute to binary array.
    # mlb = MultiLabelBinarizer(attrs)
    # attr_rec = mlb.fit_transform(list(aid_attr_dict.values()))
    # id_attr_dict = {id_: attr_rec[id_, :] for id_, pid in enumerate(filtered_authors.keys())}
    # attr = pd.DataFrame.from_dict(id_attr_dict, orient='index', columns=attrs)

    return filtered_authors, aid_attr_dict


def compute_attributes(dblp_data, cache_folder):
    aid_attr_dict = dblp_data['aid_attr_dict']
    aid_id_dict = dblp_data['aid_id_dict']

    attributes_file = join(cache_folder, "countries.attributes")
    if not os.path.exists(attributes_file):
        id_attr_list = np.array([[aid_id_dict[aid], attr] for aid, attr in aid_attr_dict.items()])
        np.savetxt(attributes_file, id_attr_list, fmt="%s", delimiter=':')


def dblp_flow(data_name, data_folder, subsample_step=8, seed=0):
    random.seed(seed)
    np.random.seed(seed)

    cache_folder = join('./cached_data/', data_name)
    dblp_data = load_dblp_data(data_name, data_folder, cache_folder, subsample_step)

    # Get the authors for which we can find a country.
    author_records, aid_attr_dict = get_author_with_country(dblp_data['author_records'])
    dblp_data['author_records'] = author_records
    dblp_data['aid_attr_dict'] = aid_attr_dict

    # Since we now have less authors, find new indexes for them.
    aid_id_dict = {aid: i for i, aid in enumerate(author_records.keys())}
    dblp_data['aid_id_dict'] = aid_id_dict
    id_aid_dict = {i: aid for i, aid in enumerate(author_records.keys())}
    dblp_data['id_aid_dict'] = id_aid_dict
    compute_edgelist(dblp_data, 'author-author', cache_folder)

    # Save attribute data.
    compute_attributes(dblp_data, cache_folder)


if __name__ == "__main__":
    data_name_ = 'ml_dm'
    dblp_flow(data_name_, join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "dblp"))
