#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <memory>
#include <cmath>
#include <sstream>
#include "omp.h"

using namespace std;


const int K = 100;
int NUSER, NITEMS;
string base_name;

map<int, float> popular_tracks;
map<int, float> popular_tags;
vector<vector<int>> tracks_in_playlist;

float gen_rand() {
    return ((float) rand() / (RAND_MAX));
}

class Track {
public:
    int id, album_id, artist_id, duration, playcount;
    vector<int> tags;
    vector<int> playlist;
    vector<int> playlist_len;

    Track(int id, int album_id, int artist_id, int duration, int playcount, vector<int> &tags, vector<int>& playlist, vector<int>& playlist_len) : id(id), album_id(album_id), artist_id(artist_id), duration(duration), playcount(playcount), tags(tags), playlist(playlist), playlist_len(playlist_len) {}
};

class Metric {
public:
    virtual float dist(const Track& t1, const Track& t2) = 0;
    virtual void print_weights(ofstream& output) = 0;
};

class LinearMetric : public Metric {
public:
    float w_artist, w_album, w_duration, w_playcount, w_tags, w_playlist, w_popularity_track;
    LinearMetric() {
        w_artist = gen_rand();
        w_album = gen_rand();
        w_duration = gen_rand();
        w_playcount = gen_rand();
        w_tags = gen_rand();
        w_playlist = gen_rand();
        w_popularity_track = gen_rand();
    }
    LinearMetric(float w_artist, float w_album, float w_duration, float w_playcount, float w_tags, float w_playlist, float w_popularity_track)
        : w_artist(w_artist), w_album(w_album), w_duration(w_duration), w_playcount(w_playcount), w_tags(w_tags), w_playlist(w_playlist), w_popularity_track(w_popularity_track)
         {
         }
    void print_weights(ofstream& output) {
        output << w_artist << " " << w_album << " " << w_duration << " " << w_playcount << " " << w_tags << " " << w_playlist << endl;
    }

    float dist(const Track& t1, const Track& t2) {
        float tags_in_common = 0, playlist_in_common = 0;
        int i = 0, j = 0;
        while (i < t1.tags.size() && j < t2.tags.size()) {
            if (t1.tags[i] == t2.tags[j]) {
                if (popular_tags.find(t1.tags[i]) != popular_tags.end())
                    tags_in_common += popular_tags[t1.tags[i]];
                i++, j++;
            }
            else if (t1.tags[i] > t2.tags[j])
                j++;
            else
                i++;
        }
        tags_in_common = (tags_in_common) / (t1.tags.size() + t2.tags.size() + 10);

        while (i < t1.playlist.size() && j < t2.playlist.size()) {
            if (t1.playlist[i] == t2.playlist[j]) {
                playlist_in_common += 4.0f/(t1.playlist_len[i] + t2.playlist_len[j] + 0.5);
                // playlist_in_common ++;
                // playlist_in_common += sqrt(100 / tracks_in_playlist[t1.playlist[i]].size() + 5);
                i++, j++;
            }
            else if (t1.playlist[i] > t2.playlist[j])
                j++;
            else
                i++;
        }
        // playlist_in_common = (playlist_in_common) / (t1.playlist.size() + t2.playlist.size() + 10);

        float duration_val = (t1.duration == -1 || t2.duration == -1) ? 0 : exp(-abs(t1.duration - t2.duration)/100000.0);
        float playcount_val = (t1.playcount == -1 || t2.playcount == -1) ? 0 : exp(-abs(t1.playcount - t2.playcount)/100.0);
        float same_artist = ((t1.artist_id == -1 || t2.artist_id == -1) ? 0 : t1.artist_id == t2.artist_id);
        float same_album = ((t1.album_id == -1 || t2.album_id == -1) ? 0 : t1.album_id == t2.album_id);

        float ret =
            ( w_artist * same_artist
            + w_album * same_album
            + w_duration * duration_val
            + w_playcount * playcount_val
            + w_tags * (tags_in_common)
            + w_playlist * (playlist_in_common)
        );
        // if (playlist_in_common > 0)
        //     cout << "[DEBUG] duration=" << duration_val << " playcount=" << playcount_val << " tags=" << tags_in_common << " playlist=" << playlist_in_common << " total=" << ret << endl;
        // float info = (same_artist + same_album + (duration_val > 0) + (playcount_val>0) + t1.tags.size()/5.0 + t2.tags.size()/5.0 + 2) / 8.0;
        // cout << info << endl;
        //return ret * info;
        return ret;
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
    TopNElements(int size) : size(size), elems(size), idxs(size) {
        idxs.assign(size, 0);
        elems.assign(size, -1);
    }
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

    void sorted(vector<float> &elements, vector<int> &indexes) {
        vector<pair<int, float>> V;
        for (int i = 0; i < size; i++)
            V.push_back({idxs[i], elems[i]});
        sort(V.begin(), V.end());

        elements.resize(size);
        indexes.resize(size);
        for (int i = 0; i < size; i++){
            elements[i] = V[i].second;
            indexes[i] = V[i].first;
        }
    }
};

















void compute_similarity(unique_ptr<Metric> &metric, vector<Track>& tracks, vector<Track>& target_tracks, vector<int>& mapping, vector<vector<float>>& similarity, vector<vector<int>>& indexes, int K = 20) {
    similarity.clear();
    indexes.clear();

    similarity.assign(tracks.size(), vector<float>());
    indexes.assign(tracks.size(), vector<int>());


    #pragma omp parallel for
    for (int i = 0; i < tracks.size(); i++) {
        auto topn = TopNElements(K);
        for (int j = 0; j < tracks.size(); j++) {
            if (i != j)
                topn.push(metric->dist(tracks[i], tracks[j]), j);
        }

        vector<float> elms;
        vector<int> idx;
        topn.sorted(elms, idx);

        similarity[i] = elms;
        indexes[i] = idx;
        // similarity[mapping[i]] = elms;
        // indexes[mapping[i]] = idx;

        if (i % 500 == 0) {
            printf("Track %d of %d for thread %d\n", i, tracks.size()*(1 + omp_get_thread_num())/omp_get_num_threads(), omp_get_thread_num());
        }
    }
}



Metric *parse_similarity_args(int argc, char *argv[]) {
    Metric *m;
    if (argc < 3)
        m = new LinearMetric();
    else if (argc == 9) {
        float w_artist, w_album, w_duration, w_playcount, w_tags, w_playlist, w_popularity_track;
        w_artist = atof(argv[1]);
        w_album = atof(argv[2]);
        w_duration = atof(argv[3]);
        w_playcount = atof(argv[4]);
        w_tags = atof(argv[5]);
        w_playlist = atof(argv[6]);
        w_popularity_track = atof(argv[7]);
        m = new LinearMetric(w_artist, w_album, w_duration, w_playcount, w_tags, w_playlist, w_popularity_track);
    }
    else {
        cout << "Usage: " << argv[0] << " <w_artist> <w_album> <w_duration> <w_playcount> <w_tags> <w_playlist> <w_popularity_track> NAME_DIR" << endl;
        exit(0);
    }
    return m;
}


void get_target_tracks(set<int> &ttracks) {
    //fstream fs;
    //fs.open(base_name + "target_tracks.txt", std::fstream::in);
    ifstream fs (base_name + "/target_tracks.txt");

    if (!fs) {
        printf("Error opening target_tracks.txt\n");
        exit(0);
    }

    ttracks.clear();

    int N;
    fs >> N;

    for (int i = 0; i < N; i++) {
        int tmp;
        fs >> tmp;
        ttracks.insert(tmp);
    }
    fs.close();
}


void read_tracks(vector<Track>& tracks, vector<Track>& target_tracks, vector<int>& mapping) {
    set<int> ttracks;
    get_target_tracks(ttracks);

    int N, id, album_id, artist_id, duration, playcount, ntags, t, nplaylist;

    ifstream input (base_name + "/tracks.txt");

    if (!input) {
        printf("Error opening tracks.txt\n");
        exit(0);
    }

    input >> N;

    for (int i = 0; i < N; i++) {
        vector<int> tags;
        vector<int> playlist;
        vector<int> playlist_len;

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
        for (int j = 0; j < nplaylist; j++) {
            int len = tracks_in_playlist[playlist[j]].size();
            // if (tracks_in_playlist.find(playlist[j]) != tracks_in_playlist.end()) {
            //     len = tracks_in_playlist[playlist[j]].size();
            // }
            playlist_len.push_back(len);
        }

        tracks.push_back(Track(id, album_id, artist_id, duration, playcount, tags, playlist, playlist_len));
        if (ttracks.find(id) != ttracks.end()) {
            target_tracks.push_back(Track(id, album_id, artist_id, duration, playcount, tags, playlist, playlist_len));
            mapping.push_back(i);
        }

    }
}

void print_similarity(ofstream& output, vector<vector<float>>& similarity, vector<vector<int>>& indexes) {
    output << NITEMS << " " << NITEMS << endl;

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

    if (!input) {
        printf("Error opening popular_tracks.txt\n");
        exit(0);
    }

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

    if (!input) {
        printf("Error opening popular_tags.txt\n");
        exit(0);
    }

    int N;
    input >> N;

    while (N--) {
        int tag_id;
        float weight;
        input >> tag_id >> weight;
        popular_tags[tag_id] = weight;
    }
}

void read_tracks_in_playlist(vector<vector<int>>& tracks_in_playlist) {
    ifstream input (base_name + "/tracks_in_playlist.txt");

    if (!input) {
        printf("Error opening tracks_in_playlist.txt\n");
        exit(0);
    }
    string sizes, rows, cols;

    getline(input, sizes);
    getline(input, rows);
    getline(input, cols);

    stringstream sizestream(sizes);
    stringstream rstream(rows);
    stringstream cstream(cols);

    sizestream >> NUSER >> NITEMS;
    tracks_in_playlist.resize(NUSER);

    int r,c;
    while (rstream >> r) {
        cstream >> c;
        tracks_in_playlist[r].push_back(c);
    }

    for (int i = 0; i < NUSER; i++)
        sort(tracks_in_playlist[i].begin(), tracks_in_playlist[i].end());
}





int sample_negative(int user, map<int, vector<int>>& urm) {
    int j = rand() % (100000);
    while (find(urm[user].begin(), urm[user].end(), j) != urm[user].end())
        j = rand() % (100000);

    return j;
}

void sample(int& u, int& i, int& j, map<int, vector<int>>& urm) {
    u = rand()%(urm.size());
    i = urm[u][rand() % (urm[u].size())];
    j = sample_negative(u, urm);
}

float predict_xuij(int u, int i, int j, map<int,vector<int>>& urm, vector<vector<float>> &S, vector<vector<int>> indexes) {
    int psi = 0, psj = 0;
    float count = 0.0;
    for (int k = 0; k < urm[u].size(); k++) {
        // Advance psi & psj
        while (psi < indexes[i].size() && urm[u][k] < indexes[i][psi])
            psi++;
        while (psj < indexes[j].size() && urm[u][k] < indexes[j][psi])
            psj++;

        if (psi < indexes[i].size() && urm[u][k] == indexes[i][psi])
            count += S[i][psi];
        if (psj < indexes[j].size() && urm[u][k] == indexes[j][psj])
            count -= S[j][psj];
    }

    return count;
}

void BPRSLIM(map<int, vector<int>>& urm, vector<vector<float>> &S, vector<vector<int>> &indexes, int iterations=1, float alpha = 0.1, float reg_positive = 0.1, float reg_negative=0.1) {
    for (int it = 0; it < iterations; it++) {
        for (int count = 0; count < 100; count++) {
            int u,i,j;
            sample(u, i, j, urm);
            cout << "sampled u=" << u << ", i=" << i << ", j=" << j << endl;

            float x_pred = predict_xuij(u, i, j, urm, S, indexes);
            float z = exp(-x_pred) * (1 + exp(-x_pred));
            cout << "x_pred=" << x_pred << ", z=" << z << endl;

            int psi = 0, psj = 0;
            for (int l = 0; l < urm[u].size(); l++) {
                int idx = urm[u][l];
                while (psi < indexes[i].size() && idx < indexes[i][psi])
                    psi++;
                while (psj < indexes[j].size() && idx < indexes[j][psj])
                    psj++;

                if (idx != i && idx == indexes[i][psi]) {
                    cout << "Updated S[" << i << "][" << idx << "] " << S[i][psi] << " --> ";
                    S[i][psi] += alpha*(z - reg_positive*S[i][psi]);
                    cout << S[i][psi] << endl;

                }

                if (idx != j && idx == indexes[j][psj]){
                    cout << "Updated S[" << j << "][" << idx << "] " << S[j][psj] << " --> ";
                    S[j][psj] += alpha*(-z - reg_negative*S[j][psj]);
                    cout << S[j][psj] << endl;
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    vector<Track> tracks;
    vector<Track> target_tracks;
    vector<int> mapping;
    vector<vector<float>> similarity;
    vector<vector<int>> indexes;

    unique_ptr<Metric> metric(parse_similarity_args(argc, argv));
    base_name = string(argv[8]);

    cout << "Reading tracks in playlist..." << endl;
    read_tracks_in_playlist(tracks_in_playlist);
    cout << "Reading tracks..." << endl;
    read_tracks(tracks, target_tracks, mapping);

    cout << "Creating similarity matrix..." << endl;
    compute_similarity(metric, tracks, target_tracks, mapping, similarity, indexes, K);

    //BPRSLIM(tracks_in_playlist, similarity, indexes);

    ofstream output;
    output.open (base_name + "/similarity.txt");

    cout << "Printing similarity matrix..." << endl;
    print_similarity(output, similarity, indexes);
}
