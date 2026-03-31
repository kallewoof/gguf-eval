/*
 * Source: https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/blob/main/convert.cpp
 * Modification: complete support for converting to/from bin/json (original code only converts from bin to json)
 */

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <exception>

#include "json.hpp"

namespace {

void serializeString(std::ostream& out, const std::string& s) {
    uint32_t n = s.size();
    out.write((char *)&n, sizeof(n));
    out.write(s.data(), n);
}
bool deserializeString(std::istream& in, std::string& s) {
    uint32_t n;
    if (!in.read((char *)&n, sizeof(n)).fail()) {
        s.resize(n);
        return !in.read((char *)s.data(), s.size()).fail();
    }
    return false;
}

struct Answers {
    std::vector<std::string> answers;
    std::vector<int>         labels;
    void serialize(std::ostream& out) const {
        if (answers.size() != labels.size()) {
            throw std::runtime_error("Inconsistent number of answers and labels");
        }
        uint32_t n = answers.size();
        out.write((char *)&n, sizeof(n));
        for (auto& a : answers) {
            serializeString(out, a);
        }
        out.write((char *)labels.data(), labels.size()*sizeof(int));
    }
    bool deserialize(std::istream& in) {
        int n;
        in.read((char *)&n, sizeof(n));
        if (in.fail() || n < 0) {
            return false;
        }
        answers.resize(n);
        labels.resize(n);
        for (auto& a : answers) {
            if (!deserializeString(in, a)) return false;
        }
        in.read((char *)labels.data(), n*sizeof(int));
        return !in.fail();
    }
    nlohmann::json toJson() const {
        nlohmann::json o = nlohmann::json::object();
        o["answers"] = answers;
        o["labels"]  = labels;
        return o;
    }
};

struct MultiplChoice {
    std::string question;
    Answers singleCorrect;
    Answers multipleCorrect;

    static MultiplChoice from_json(const nlohmann::json& item) {
        MultiplChoice m;

        /*
         PIQA: question="goal", single_correct.answers="solN", single_correct.labels=[1 if label matches index else 0]
         {
             "goal":"How do I ready a guinea pig cage for its new occupants?",
             "sol1":"Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish.",
             "sol2":"Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish.",
             "label":0
         }
         */
        if (item.contains("goal") && item.contains("sol1") && item.contains("label")) {
            m.question = item.value("goal", "");
            auto& answers = m.singleCorrect.answers;
            auto& labels = m.singleCorrect.labels;
            int label = 1 + item.value("label", -99);
            for (int i = 1; item.contains("sol" + std::to_string(i)); ++i) {
                answers.push_back(item.value("sol" + std::to_string(i), ""));
                labels.push_back(label == i);
            }
            // const auto& example = m.singleCorrect.answers[0];
            // printf("Question: %s. Answer example: %s\n", m.question.c_str(), example.c_str());
            return m;
        }

        /*
         BOOLQ:
         {
            "question":"does ethanol take more energy make that produces",
            "answer":false,
            "passage":"All biomass goes through at least some of[...]"
         }
         */
        if (item.contains("question") && item.contains("answer") && item.contains("passage")) {
            // We reformat this as:
            // question = "Given the following passage, answer the provided question.\n\nPassage: {passage}\n\nQuestion: {question}\n\n"
            // singleCorrect.answers = ["Answer: yes", "Answer: no"]
            // singleCorrect.labels = [1 if true else 0, 1 if false else 0]
            m.question = "Given the following passage, answer the provided question.\n\nPassage: " + item.value("passage", "") + "\n\nQuestion: " + item.value("question", "") + "\n\n";
            m.singleCorrect.answers.push_back("Answer: yes");
            m.singleCorrect.answers.push_back("Answer: no");
            bool answer = item.value("answer", true);
            m.singleCorrect.labels.push_back(int(answer));
            m.singleCorrect.labels.push_back(int(!answer));
            const auto& example = m.singleCorrect.answers[0];
            printf("Question: %s. Answer example: %s\n", m.question.c_str(), example.c_str());
            return m;
        }

        // Fallback to IK format
        m.question = item.value("question", "");
        m.singleCorrect.answers = item.value("single_correct", nlohmann::json::object()).value("answers", std::vector<std::string>());
        m.singleCorrect.labels  = item.value("single_correct", nlohmann::json::object()).value("labels", std::vector<int>());
        m.multipleCorrect.answers = item.value("multiple_correct", nlohmann::json::object()).value("answers", std::vector<std::string>());
        m.multipleCorrect.labels  = item.value("multiple_correct", nlohmann::json::object()).value("labels", std::vector<int>());

        return m;
    }

    void serialize(std::ostream& out) const {
        serializeString(out, question);
        singleCorrect.serialize(out);
        multipleCorrect.serialize(out);
    }
    bool deserialize(std::istream& in) {
        if (!deserializeString(in, question)) return false;
        return singleCorrect.deserialize(in) && multipleCorrect.deserialize(in);
    }
    nlohmann::json toJson() const {
        nlohmann::json o = nlohmann::json::object();
        o["question"] = question;
        o["single_correct"  ]  = singleCorrect.toJson();
        o["multiple_correct"]  = multipleCorrect.toJson();
        return o;
    }
    static nlohmann::json toJson(const std::vector<MultiplChoice>& data) {
        nlohmann::json o = nlohmann::json::array();
        for (auto& d : data) o.push_back(d.toJson());
        return o;
    }
    static std::vector<MultiplChoice> loadFromStream(std::istream& in) {
        uint32_t n;
        if (in.read((char *)&n, sizeof(n)).fail()) {
            printf("%s: failed reading number of entries\n", __func__);
            return {};
        }
        in.seekg(n*sizeof(uint32_t), std::ios::cur); // skip positions
        std::vector<MultiplChoice> result(n);
        int i = 0;
        for (auto& r : result) {
            ++i;
            if (!r.deserialize(in)) {
                printf("%s: failed reading data at question %d\n", __func__, i);
                return {};
            }
        }
        return result;
    }
    static std::vector<MultiplChoice> loadFromFile(const char* fileName) {
        std::ifstream in(fileName, std::ios::binary);
        if (!in) {
            printf("%s: failed to open %s\n", __func__, fileName);
            return {};
        }
        return loadFromStream(in);
    }
    static std::vector<MultiplChoice> loadFromJSONFile(const char* fileName) {
        std::ifstream in(fileName);
        if (!in) {
            printf("%s: failed to open %s\n", __func__, fileName);
            return {};
        }
        nlohmann::json j;
        try {
            in >> j;
        } catch (const nlohmann::json::parse_error& e) {
            printf("%s: failed to parse JSON from %s: %s\n", __func__, fileName, e.what());
            return {};
        }
        if (!j.is_array()) {
            printf("%s: JSON data is not an array in %s\n", __func__, fileName);
            return {};
        }
        std::vector<MultiplChoice> result;
        for (const auto& item : j) {
            result.push_back(from_json(item));
        }
        return result;
    }
    static std::vector<MultiplChoice> loadFromJSONLFile(const char* fileName) {
        std::ifstream in(fileName);
        if (!in) {
            printf("%s: failed to open %s\n", __func__, fileName);
            return {};
        }
        nlohmann::json j;
        std::vector<nlohmann::json> entries;
        while (!in.eof()) {
            try {
                in >> j;
            } catch (const nlohmann::json::parse_error& e) {
                if (!in.eof() || entries.size() == 0) {
                    printf("%s: failed to parse JSON from %s: %s\n", __func__, fileName, e.what());
                    return {};
                }
            }
            entries.push_back(j);
        }
        std::vector<MultiplChoice> result;
        for (const auto& item : entries) {
            result.push_back(from_json(item));
        }
        return result;
    }
    static void serialize(std::ostream& out, const std::vector<MultiplChoice>& data) {
        uint32_t n = data.size();
        out.write((char *)&n, sizeof(n));
        if (data.empty()) return;
        std::vector<uint32_t> pos(data.size(), 0);
        out.write((char *)pos.data(), pos.size()*sizeof(pos[0]));
        int i = 0;
        for (auto& d : data) {
            pos[i++] = out.tellp();
            d.serialize(out);
        }
        out.seekp(sizeof(n), std::ios::beg);
        out.write((char *)pos.data(), pos.size()*sizeof(pos[0]));
    }
    static void serialize(const char* fileName, const std::vector<MultiplChoice>& data) {
        std::ofstream out(fileName, std::ios::binary);
        if (!out) {
            printf("%s: failed to open %s for writing\n", __func__, fileName);
            return;
        }
        serialize(out, data);
    }
};
}

bool ends_with(const std::string& fullString, const std::string& ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}


int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s [input_file] [output_file]\n", argv[0]);
        return 1;
    }

    std::vector<MultiplChoice> data;
    if (ends_with(argv[1], ".bin")) {
        data = MultiplChoice::loadFromFile(argv[1]);
    } else if (ends_with(argv[1], ".json")) {
        data = MultiplChoice::loadFromJSONFile(argv[1]);
    } else if (ends_with(argv[1], ".jsonl")) {
        data = MultiplChoice::loadFromJSONLFile(argv[1]);
    } else {
        printf("Unsupported file format. Please provide a .bin or .json file.\n");
        return 1;
    }
    printf("Loaded %zu datasets from %s\n", data.size(), argv[1]);

    auto ofile = argc > 2 ? std::string{argv[2]} : std::string{argv[1]} + ".json";
    std::ofstream out(ofile.c_str());

    if (ends_with(argv[2], ".bin")) {
        MultiplChoice::serialize(out, data);
    } else if (ends_with(argv[2], ".json")) {
        auto json = MultiplChoice::toJson(data);
        out << json << std::endl;
    } else {
        printf("Unsupported output format. Please provide a .bin or .json file.\n");
        return 1;
    }

    return 0;
}
