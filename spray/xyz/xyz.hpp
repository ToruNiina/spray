#ifndef SPRAY_XYZ_XYZ_HPP
#define SPRAY_XYZ_XYZ_HPP
#include <spray/util/log.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>

namespace spray
{
namespace xyz
{

struct particle
{
    std::string         name;
    std::array<float, 3> vec;
};

struct snapshot
{
    std::string           comment;
    std::vector<particle> particles;
};

struct reader
{
    reader(const std::string& filename)
        : filename_(filename)
    {
        std::ifstream ifs(filename_);
        if(!ifs.good())
        {
            spray::log(spray::log_level::error,
                       "file open error \"", this->filename_, "\"");
            throw std::runtime_error("Error: file open error: " + filename_);
        }
        this->snapshots_.push_back(ifs.tellg());
    }

    snapshot read_snapshot(const std::size_t i)
    {
        std::ifstream ifs(filename_);
        if(i < snapshots_.size())
        {
            ifs.seekg(snapshots_.at(i));
        }
        else // need to search the i-th snapshot
        {
            ifs.seekg(snapshots_.back());
            const std::size_t diff = i+1 - snapshots_.size();
            for(std::size_t j=0; j<diff; ++j)
            {
                this->skip_snapshot(ifs);
            }
        }
        return read_snapshot_impl(ifs);
    }

  private:

    snapshot read_snapshot_impl(std::ifstream& ifs)
    {
        std::string number, comment;
        std::getline(ifs, number);  // number of particles in the snapshot
        std::getline(ifs, comment); // comment.

        std::istringstream iss(number);
        std::size_t num;
        iss >> num;
        if(iss.fail()) // check `num` is successfully loaded
        {
            const auto ln = this->count_number(ifs);
            spray::log(spray::log_level::error, "file ", this->filename_,
                       " line ", ln, ": expected integer variable, got ",
                       number);
            std::exit(EXIT_FAILURE);
        }

        snapshot ss;
        ss.comment = comment;
        ss.particles.reserve(num);
        for(std::size_t i=0; i<num; ++i)
        {
            std::string line;
            std::getline(ifs, line);
            std::istringstream iss(line);

            particle p;
            iss >> p.name >> p.vec[0] >> p.vec[1] >> p.vec[2];
            ss.particles.push_back(p);
        }
        return ss;
    }

    void skip_snapshot(std::ifstream& ifs)
    {
        std::string number, comment;
        std::getline(ifs, number);  // number of particles in the snapshot
        std::getline(ifs, comment); // comment.

        std::istringstream iss(number);
        std::size_t num;
        iss >> num;
        if(iss.fail()) // check `num` is successfully loaded
        {
            const auto ln = this->count_number(ifs);
            spray::log(spray::log_level::error, "file ", this->filename_,
                       " line ", ln, ": expected integer variable, got ",
                       number);
            std::exit(EXIT_FAILURE);
        }
        for(std::size_t i=0; i<num; ++i)
        {
            std::getline(ifs, comment);
        }
        // assign the starting point of the next snapshot
        snapshots_.push_back(ifs.tellg());
        return;
    }

    std::size_t count_number(std::ifstream& ifs)
    {
        const auto pos = ifs.tellg();
        ifs.seekg(0);
        std::size_t num = 0;
        while(ifs.tellg() < pos)
        {
            ++num;
            std::string line;
            std::getline(ifs, line);
        }
        return num;
    }

  private:
    std::vector<std::ifstream::pos_type> snapshots_;
    std::string filename_;
};

} // xyz
} // spray
#endif// SPRAY_XYZ_XYZ_HPP
