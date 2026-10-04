#include <heongpu/util/serializer.h>
#include <algorithm>
#include <array>
#include <limits>

namespace heongpu
{
    namespace serializer
    {

        std::vector<uint8_t> to_buffer(const std::stringstream& ss)
        {
            const std::string& str = ss.str();
            return {str.begin(), str.end()};
        }

        void from_buffer(std::stringstream& ss,
                         const std::vector<uint8_t>& buffer)
        {
            ss.str(std::string(buffer.begin(), buffer.end()));
        }

        std::vector<uint8_t> compress(const std::vector<uint8_t>& data)
        {
            uLongf bound = compressBound(data.size());
            std::vector<uint8_t> out(bound);

            if (::compress(out.data(), &bound, data.data(), data.size()) !=
                Z_OK)
            {
                throw std::runtime_error("Zlib compression failed");
            }

            out.resize(bound);
            return out;
        }

        std::vector<uint8_t> decompress(const std::vector<uint8_t>& data)
        {
            if (data.empty())
                throw std::runtime_error("Empty zlib input");
            z_stream stream{};
            if (::inflateInit(&stream) != Z_OK)
                throw std::runtime_error("Zlib initialization failed");
            struct InflateCleanup
            {
                z_stream* stream;
                ~InflateCleanup() { ::inflateEnd(stream); }
            } cleanup{&stream};
            std::array<uint8_t, 64 * 1024> chunk;
            std::vector<uint8_t> out;
            size_t offset = 0;
            for (;;)
            {
                if (stream.avail_in == 0 && offset < data.size())
                {
                    const size_t count =
                        std::min(data.size() - offset,
                                 size_t(std::numeric_limits<uInt>::max()));
                    stream.next_in = const_cast<Bytef*>(data.data() + offset);
                    stream.avail_in = static_cast<uInt>(count);
                    offset += count;
                }
                stream.next_out = chunk.data();
                stream.avail_out = chunk.size();
                const int result = ::inflate(&stream, Z_NO_FLUSH);
                if (result != Z_OK && result != Z_STREAM_END)
                    throw std::runtime_error("Zlib decompression failed");
                const size_t count = chunk.size() - stream.avail_out;
                out.insert(out.end(), chunk.begin(), chunk.begin() + count);
                if (result == Z_STREAM_END)
                    break;
                if (count == 0 && stream.avail_in == 0 && offset == data.size())
                    throw std::runtime_error("Truncated zlib input");
            }
            return out;
        }

    } // namespace serializer
} // namespace heongpu
