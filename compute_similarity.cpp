#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include "omp.h"

using namespace std;


const int K = 20;
string base_name;


float gen_rand() {
    return ((float) rand() / (RAND_MAX));
}

class Track {
public:
    int id, album_id, artist_id, duration, playcount;
    vector<int> tags;
    vector<int> playlist;

    Track(int id, int album_id, int artist_id, int duration, int playcount, vector<int> &tags, vector<int>& playlist) : id(id), album_id(album_id), artist_id(artist_id), duration(duration), playcount(playcount), tags(tags), playlist(playlist) {}
};

class Metric {
public:
    virtual float dist(const Track& t1, const Track& t2) = 0;
    virtual void print_weights(ofstream& output) = 0;
};

class LinearMetric : public Metric {
public:
    float w_artist, w_album, w_duration, w_playcount, w_tags, w_playlist;
    LinearMetric() {
        w_artist = gen_rand();
        w_album = gen_rand();
        w_duration = gen_rand();
        w_playcount = gen_rand();
        w_tags = gen_rand();
        w_playlist = gen_rand();
    }
    LinearMetric(float w_artist, float w_album, float w_duration, float w_playcount, float w_tags, float w_playlist)
        : w_artist(w_artist), w_album(w_album), w_duration(w_duration), w_playcount(w_playcount), w_tags(w_tags), w_playlist(w_playlist)
         {
         }
    void print_weights(ofstream& output) {
        output << w_artist << " " << w_album << " " << w_duration << " " << w_playcount << " " << w_tags << " " << w_playlist << endl;
    }

    float dist(const Track& t1, const Track& t2) {
        int tags_in_common = 0, playlist_in_common = 0;
        int i = 0, j = 0;
        while (i < t1.tags.size() && j < t2.tags.size()) {
            if (t1.tags[i] == t2.tags[j])
                tags_in_common++, i++, j++;
            else if (t1.tags[i] > t2.tags[j])
                j++;
            else
                i++;
        }
        while (i < t1.playlist.size() && j < t2.playlist.size()) {
            if (t1.playlist[i] == t2.playlist[j])
                playlist_in_common++, i++, j++;
            else if (t1.playlist[i] > t2.playlist[j])
                j++;
            else
                i++;
        }

        return
            ( w_artist * (t1.artist_id == t2.artist_id)
            + w_album * (t1.album_id == t2.album_id)
            + w_duration * (abs(t1.duration - t2.duration))
            + w_playcount * (abs(t1.playcount - t2.playcount))
            + w_tags * (tags_in_common)
            + w_playlist * (playlist_in_common)
        );
        //    / (w_artist + w_album + w_duration + w_playcount + w_tags + 1e06);
    }
};

class Playlist {
public:
    int id;
    vector<int> track_ids;
    Playlist(int id, vector<int>& track_ids) : id(id), track_ids(track_ids) {}
};

class TopNElements {
public:
    int size;
    vector<float> elems;
    vector<int> idxs;
    TopNElements(int size) : size(size), elems(size), idxs(size) {}
    void push(float e, int index) {
        if (e <= elems[size-1])
            return;
        int i = size -1;
        while (i > 0 && e > elems[i-1]) {
            elems[i] = elems[i-1];
            idxs[i] = idxs[i-1];
            i--;
        }
        elems[i] = e;
        idxs[i] = index;
    }
};

void compute_similarity(Metric* metric, vector<Track>& tracks, vector<vector<float>>& similarity, vector<vector<int>>& indexes, int K = 30) {
    similarity.clear();
    indexes.clear();

    similarity.assign(tracks.size(), vector<float>());
    indexes.assign(tracks.size(), vector<int>());

    #pragma omp parallel for
    for (int i = 0; i < tracks.size(); i++) {
        auto topn = TopNElements(K);
        for (int j = 0; j < tracks.size(); j++) {
            topn.push(metric->dist(tracks[i], tracks[j]), j);
        }

        vector<float> elms(topn.elems);
        vector<int> idx(topn.idxs);

        similarity[i] = elms;
        indexes[i] = idx;

        if (i % 500 == 0) {
            printf("Track %d of %d for thread %d\n", i, tracks.size()*(1 + omp_get_thread_num())/omp_get_num_threads(), omp_get_thread_num());
        }
    }
}



Metric *parse_similarity_args(int argc, char *argv[]) {
    Metric *m;
    if (argc < 3)
        m = new LinearMetric();
    else if (argc == 8) {
        float w_artist, w_album, w_duration, w_playcount, w_tags, w_playlist;
        w_artist = atof(argv[1]);
        w_album = atof(argv[2]);
        w_duration = atof(argv[3]);
        w_playcount = atof(argv[4]);
        w_tags = atof(argv[5]);
        w_playlist = atof(argv[6]);
        m = new LinearMetric(w_artist, w_album, w_duration, w_playcount, w_tags, w_playlist);
    }
    else {
        cout << "Usage: " << argv[0] << " <w_artist> <w_album> <w_duration> <w_playcount> <w_tags>" << endl;
        exit(0);
    }
    return m;
}

void read_tracks(vector<Track>& tracks) {
    int N, id, album_id, artist_id, duration, playcount, ntags, t, nplaylist;

    ifstream input (base_name + "/input.txt");
    input >> N;

    for (int i = 0; i < N; i++) {
        vector<int> tags;
        vector<int> playlist;

        input >> id >> album_id >> artist_id >> duration >> playcount >> ntags;

        for (int j = 0; j < ntags; j++) {
            input >> t;
            tags.push_back(t);
        }
        input >> nplaylist;
        for (int j = 0; j < nplaylist; j++) {
            input >> t;
            playlist.push_back(t);
        }
        sort(tags.begin(), tags.end());
        sort(playlist.begin(), playlist.end());
        tracks.push_back(Track(id, album_id, artist_id, duration, playcount, tags, playlist));
    }
}

void print_similarity(ofstream& output, vector<vector<float>>& similarity, vector<vector<int>>& indexes) {
    // Print rows
    for (int i = 0; i < similarity.size(); i++)
        for (int j = 0; j < similarity[i].size(); j++)
            output << i << " ";
    output << endl;

    // Print cols
    for (int i = 0; i < similarity.size(); i++)
        for (int j = 0; j < similarity[i].size(); j++) {
            output << indexes[i][j] << " ";
        }
    output << endl;

    // Print data
    for (int i = 0; i < similarity.size(); i++) {
        for (int j = 0; j < similarity[i].size(); j++)
            output << similarity[i][j] << " ";
    }
    output << endl;
}

void read_popular_tracks(map<int, float>& popular_tracks) {
    ifstream input (base_name + "/popular_tracks.txt");

    int N;
    input >> N;

    while (N--) {
        int track_id;
        float weight;
        input >> track_id >> weight;
        popular_tracks[track_id] = weight;
    }
}

void read_popular_tags(map<int, float>& popular_tags) {
    ifstream input (base_name + "/popular_tags.txt");

    int N;
    input >> N;

    while (N--) {
        int tag_id;
        float weight;
        input >> tag_id >> weight;
        popular_tags[tag_id] = weight;
    }
}

int main(int argc, char *argv[]) {
    ios_base::sync_with_stdio(false);
    srand(time(NULL));

    vector<Track> tracks;
    vector<vector<float>> similarity;
    vector<vector<int>> indexes;
    map<int, float> popular_tracks;
    map<int, float> popular_tags;

    Metric *metric = parse_similarity_args(argc, argv);
    base_name = string(argv[7]);
    
    cout << "Reading tracks..." << endl;
    read_tracks(tracks);
    cout << "Reading popular tracks..." << endl;
    read_popular_tracks(popular_tracks);
    cout << "Reading popular tags..." << endl;
    read_popular_tags(popular_tags);

    cout << "Creating similarity matrix..." << endl;
    compute_similarity(metric, tracks, similarity, indexes, K);

    ofstream output;
    output.open (base_name + "/similarity.txt");

    metric->print_weights(output);
    cout << "Printing similarity matrix..." << endl;
    print_similarity(output, similarity, indexes);
}
