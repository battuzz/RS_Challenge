#include <iostream>
#include <vector>

using namespace std;

class Track {
public:
    int id, album_id, artist_id, duration, playcount;
    vector<int> tags;

    Track(int id, int album_id, int artist_id, int duration, int playcount, vector<int> &tags) : id(id), album_id(album_id), artist_id(artist_id), duration(duration), playcount(playcount), tags(tags) {}

    inline float dist(const Track& t1) {
        return abs(duration / 10 + album_id * t1.album_id - playcount * id + 24);
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

void compute_similarity(vector<Track>& tracks, vector<vector<float>>& similarity, vector<vector<int>>& indexes, int K = 20) {
    similarity.clear();
    indexes.clear();
    for (int i = 0; i < tracks.size(); i++) {
        auto topn = TopNElements(K);
        for (int j = 0; j < tracks.size(); j++) {
            topn.push(tracks[i].dist(tracks[j]), j);
        }


        similarity.push_back(topn.elems);
        indexes.push_back(topn.idxs);
    }
}


int main() {
    ios_base::sync_with_stdio(false);
    int N;
    int id, album_id, artist_id, duration, playcount, ntags, t;

    cin >> N;
    vector<Track> tracks;

    for (int i = 0; i < N; i++) {
        vector<int> tags;

        cin >> id >> album_id >> artist_id >> duration >> playcount >> ntags;
        for (int j = 0; j < ntags; j++) {
            cin >> t;
            tags.push_back(t);
        }
        tracks.push_back(Track(id, album_id, artist_id, duration, playcount, tags));
    }

    vector<vector<float>> similarity;
    vector<vector<int>> indexes;

    compute_similarity(tracks, similarity, indexes);

    for (int i = 0; i < similarity.size(); i++) {
        for (int j = 0; j < similarity[i].size(); j++)
            cout << similarity[i][j] << " " << indexes[i][j] << " ";
        cout << endl;
    }
}
