// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/util/util.cuh>
#include <heongpu/util/secstdparams.h>
#include <algorithm>
#include <map>

namespace heongpu
{
    namespace detail
    {
        const char* name(sec_level_type level)
        {
            switch (level)
            {
                case sec_level_type::none:
                    return "none";
                case sec_level_type::sec128:
                    return "sec128";
                case sec_level_type::sec192:
                    return "sec192";
                case sec_level_type::sec256:
                    return "sec256";
            }
            return "unknown";
        }

        const char* name(storage_type type)
        {
            switch (type)
            {
                case storage_type::host:
                    return "host";
                case storage_type::device:
                    return "device";
            }
            return "unknown";
        }

        const char* name(encoding type)
        {
            switch (type)
            {
                case encoding::slot:
                    return "slot";
                case encoding::coefficient:
                    return "coefficient";
            }
            return "unknown";
        }

        const char* name(rescale_mode mode)
        {
            switch (mode)
            {
                case rescale_mode::manual:
                    return "manual";
                case rescale_mode::automatic:
                    return "automatic";
            }
            return "unknown";
        }
    } // namespace detail

    std::uint64_t context_identity(int scheme, int ring_size,
                                   const std::vector<Modulus64>& primes,
                                   Data64 plain_modulus)
    {
        // 64-bit FNV-1a over the little-endian bytes of each parameter.
        std::uint64_t hash = 0xcbf29ce484222325ULL;
        auto add = [&](std::uint64_t value)
        {
            for (int i = 0; i < 8; i++)
            {
                hash ^= (value >> (8 * i)) & 0xFF;
                hash *= 0x100000001b3ULL;
            }
        };
        add(static_cast<std::uint64_t>(scheme));
        add(static_cast<std::uint64_t>(ring_size));
        add(primes.size());
        for (const Modulus64& prime : primes)
        {
            add(prime.value);
        }
        add(plain_modulus);
        return (hash == 0) ? 1 : hash;
    }

    bool coefficient_validator(const std::vector<int>& log_Q_bases_bit_sizes,
                               const std::vector<int>& log_P_bases_bit_sizes)
    {
        int P_size = log_P_bases_bit_sizes.size();

        int total_P_modulus = 0; // TODO: calculate it with prod
        for (int i = 0; i < P_size; i++)
        {
            total_P_modulus = total_P_modulus + log_P_bases_bit_sizes[i];
        }

        int Q_size = log_Q_bases_bit_sizes.size();

        int remainder = Q_size % P_size;
        int quotient = Q_size / P_size;

        int counter = 0;
        for (int i = 0; i < quotient; i++)
        {
            int pair_sum = 0;
            for (int j = 0; j < P_size; j++)
            {
                pair_sum = pair_sum + log_Q_bases_bit_sizes[counter];
                counter++;
            }

            if (pair_sum > total_P_modulus)
            {
                return false;
            }
        }

        int pair_sum = 0;
        for (int j = 0; j < remainder; j++)
        {
            pair_sum = pair_sum + log_Q_bases_bit_sizes[counter];
            counter++;
        }

        if (pair_sum > total_P_modulus)
        {
            return false;
        }

        return true;
    }

    void
    validate_key_switching_digits(const std::vector<int>& log_Q_bases_bit_sizes,
                                  const std::vector<int>& log_P_bases_bit_sizes)
    {
        const int P_size = static_cast<int>(log_P_bases_bit_sizes.size());
        int P_bits = 0;
        for (int bits : log_P_bases_bit_sizes)
        {
            P_bits += bits;
        }

        const int Q_size = static_cast<int>(log_Q_bases_bit_sizes.size());
        for (int first = 0; first < Q_size; first += P_size)
        {
            const int last = std::min(first + P_size, Q_size) - 1;
            int digit_bits = 0;
            for (int i = first; i <= last; i++)
            {
                digit_bits += log_Q_bases_bit_sizes[i];
            }
            if (digit_bits > P_bits)
            {
                throw ParameterError(error_message(
                    "The key switching digit of the Q primes ", first, " to ",
                    last, " has ", digit_bits, " bits, more than the ", P_bits,
                    " bits of P! Key switching splits Q into digits of as "
                    "many primes as P has (",
                    P_size,
                    "), and P needs at least as many bits as each "
                    "digit."));
            }
        }
    }

    Data64 extendedGCD(Data64 a, Data64 b, Data64& x, Data64& y)
    {
        if (a == 0)
        {
            x = 0;
            y = 1;
            return b;
        }

        Data64 x1, y1;
        Data64 gcd = extendedGCD(b % a, a, x1, y1);

        x = y1 - (b / a) * x1;
        y = x1;

        return gcd;
    }

    Data64 modInverse(Data64 a, Data64 m)
    {
        Data64 x, y;
        Data64 gcd = extendedGCD(a, m, x, y);

        if (gcd != 1)
        {
            // Modular inverse does not exist
            return 0;
        }
        else
        {
            // Ensure the result is positive
            Data64 result = (x % m + m) % m;
            return result;
        }
    }

    int calculate_bit_size(Data64 input)
    {
        return (int) log2(input) + 1;
    }

    bool is_power_of_two(size_t number)
    {
        return (number > 0) && ((number & (number - 1)) == 0);
    }

    int calculate_bit_count(Data64 number)
    {
        return log2(number) + 1;
    }

    int calculate_big_integer_bit_count(Data64* number, int word_count)
    {
        int size = word_count;
        for (int i = (word_count - 1); i > (-1); i--)
        {
            if (number[i] == 0)
            {
                size--;
            }
            else
            {
                break;
            }
        }

        return ((size - 1) * 64) + calculate_bit_count(number[size - 1]);
    }

    namespace
    {
        // Exact for every 64-bit modulus, unlike the Barrett reduction of
        // Modulus64.
        Data64 mult_mod(Data64 a, Data64 b, Data64 modulus)
        {
            return static_cast<Data64>(static_cast<unsigned __int128>(a) * b %
                                       modulus);
        }

        Data64 exp_mod(Data64 base, Data64 exponent, Data64 modulus)
        {
            Data64 result = 1 % modulus;
            base %= modulus;
            while (exponent > 0)
            {
                if (exponent & 1)
                {
                    result = mult_mod(result, base, modulus);
                }
                base = mult_mod(base, base, modulus);
                exponent >>= 1;
            }
            return result;
        }
    } // namespace

    bool miller_rabin(const Data64& value, size_t num_rounds)
    {
        Data64 d = value - 1;
        Data64 r = 0;
        while (0 == (d & 0x1)) // #true while the last bit of r is zero
        {
            d >>= 1;
            r++;
        }
        if (r == 0)
        {
            return false;
        }

        // The first rounds use the twelve smallest primes as bases, which
        // decide every 64-bit value without error; further rounds use random
        // bases.
        static const Data64 bases[] = {2,  3,  5,  7,  11, 13,
                                       17, 19, 23, 29, 31, 37};
        constexpr size_t base_count = sizeof(bases) / sizeof(bases[0]);
        std::unique_ptr<std::random_device> rand;
        std::uniform_int_distribution<Data64> dist(3, value - 1);
        for (size_t i = 0; i < num_rounds; i++)
        {
            Data64 a;
            if (i < base_count)
            {
                a = bases[i] % value;
                if (a == 0)
                {
                    continue;
                }
            }
            else
            {
                if (!rand)
                {
                    rand = std::make_unique<std::random_device>();
                }
                a = dist(*rand);
            }
            Data64 x = exp_mod(a, d, value);
            if (x == 1 || x == value - 1)
            {
                continue;
            }
            // value is a probable prime if x becomes -1 within r - 1
            // squarings.
            for (Data64 count = 1; (count < r) && (x != value - 1); count++)
            {
                x = mult_mod(x, x, value);
            }
            if (x != value - 1)
            {
                return false;
            }
        }
        return true;
    }

    bool is_prime(const Data64& value)
    {
        // Twelve rounds with the fixed bases of miller_rabin decide every
        // 64-bit value.
        size_t num_rounds = 12;

        // First check the prime under 1000.
        static const std::vector<Data64> low_primes = {
            3ULL,   5ULL,   7ULL,   11ULL,  13ULL,  17ULL,  19ULL,  23ULL,
            29ULL,  31ULL,  37ULL,  41ULL,  43ULL,  47ULL,  53ULL,  59ULL,
            61ULL,  67ULL,  71ULL,  73ULL,  79ULL,  83ULL,  89ULL,  97ULL,
            101ULL, 103ULL, 107ULL, 109ULL, 113ULL, 127ULL, 131ULL, 137ULL,
            139ULL, 149ULL, 151ULL, 157ULL, 163ULL, 167ULL, 173ULL, 179ULL,
            181ULL, 191ULL, 193ULL, 197ULL, 199ULL, 211ULL, 223ULL, 227ULL,
            229ULL, 233ULL, 239ULL, 241ULL, 251ULL, 257ULL, 263ULL, 269ULL,
            271ULL, 277ULL, 281ULL, 283ULL, 293ULL, 307ULL, 311ULL, 313ULL,
            317ULL, 331ULL, 337ULL, 347ULL, 349ULL, 353ULL, 359ULL, 367ULL,
            373ULL, 379ULL, 383ULL, 389ULL, 397ULL, 401ULL, 409ULL, 419ULL,
            421ULL, 431ULL, 433ULL, 439ULL, 443ULL, 449ULL, 457ULL, 461ULL,
            463ULL, 467ULL, 479ULL, 487ULL, 491ULL, 499ULL, 503ULL, 509ULL,
            521ULL, 523ULL, 541ULL, 547ULL, 557ULL, 563ULL, 569ULL, 571ULL,
            577ULL, 587ULL, 593ULL, 599ULL, 601ULL, 607ULL, 613ULL, 617ULL,
            619ULL, 631ULL, 641ULL, 643ULL, 647ULL, 653ULL, 659ULL, 661ULL,
            673ULL, 677ULL, 683ULL, 691ULL, 701ULL, 709ULL, 719ULL, 727ULL,
            733ULL, 739ULL, 743ULL, 751ULL, 757ULL, 761ULL, 769ULL, 773ULL,
            787ULL, 797ULL, 809ULL, 811ULL, 821ULL, 823ULL, 827ULL, 829ULL,
            839ULL, 853ULL, 857ULL, 859ULL, 863ULL, 877ULL, 881ULL, 883ULL,
            887ULL, 907ULL, 911ULL, 919ULL, 929ULL, 937ULL, 941ULL, 947ULL,
            953ULL, 967ULL, 971ULL, 977ULL, 983ULL, 991ULL, 997ULL};

        if (value >= 3)
        {
            if ((value & 0x1) != 0)
            {
                for (int i = 0; i < low_primes.size(); i++)
                {
                    if (value == low_primes[i])
                    {
                        return true;
                    }
                    if ((value % low_primes[i]) == 0)
                    {
                        return false;
                    }
                }

                return miller_rabin(value, num_rounds);
            }
        }

        return false;
    }

    std::vector<Data64> generate_proper_primes(Data64 factor, int bit_size,
                                               size_t count)
    {
        std::vector<Data64> destination;

        // Start with (2^bit_size - 1) / factor * factor + 1
        Data64 value = ((Data64(0x1) << bit_size) - 1) / factor * factor + 1;

        Data64 lower_bound = Data64(0x1) << (bit_size - 1);
        while (count > 0 && value > lower_bound)
        {
            if (is_prime(value))
            {
                destination.emplace_back(std::move(value));
                count--;
            }
            value -= factor;
        }
        if (count > 0)
        {
            throw std::logic_error("failed to find enough qualifying primes");
        }
        return destination;
    }

    std::vector<Modulus64>
    generate_primes(size_t poly_modulus_degree,
                    const std::vector<int> prime_bit_sizes)
    {
        std::vector<Modulus64> prime_vector_;
        std::unordered_map<int, size_t> count_table;
        std::unordered_map<int, std::vector<Data64>> prime_table;
        for (int size : prime_bit_sizes)
        {
            if ((size > MAX_USER_DEFINED_MOD_BIT_COUNT) ||
                (size < MIN_USER_DEFINED_MOD_BIT_COUNT))
            {
                throw std::logic_error("invalid modulus bit size");
            }

            ++count_table[size];
        }

        Data64 factor = Data64(2) * Data64(poly_modulus_degree);
        for (const auto& table_elt : count_table)
        {
            prime_table[table_elt.first] = generate_proper_primes(
                factor, table_elt.first, table_elt.second);
        }

        for (int size : prime_bit_sizes)
        {
            prime_vector_.emplace_back(Modulus64(prime_table[size].back()));
            prime_table[size].pop_back();
        }

        return prime_vector_;
    }

    std::vector<Modulus64> generate_internal_primes(size_t poly_modulus_degree,
                                                    const int prime_count)
    {
        std::vector<Modulus64> all_primes;

        std::vector<int> prime_bit_sizes;
        for (int i = 0; i < prime_count; i++)
        {
            prime_bit_sizes.push_back(MAX_MOD_BIT_COUNT);
        }

        std::unordered_map<int, size_t> count_table;
        std::unordered_map<int, std::vector<Data64>> prime_table;
        for (int size : prime_bit_sizes)
        {
            ++count_table[size];
        }

        Data64 factor = Data64(2) * Data64(poly_modulus_degree);
        for (const auto& table_elt : count_table)
        {
            prime_table[table_elt.first] = generate_proper_primes(
                factor, table_elt.first, table_elt.second);
        }

        for (int size : prime_bit_sizes)
        {
            all_primes.emplace_back(Modulus64(prime_table[size].back()));
            prime_table[size].pop_back();
        }

        return all_primes;
    }

    int max_coeff_bit_count(sec_level_type level, size_t poly_modulus_degree)
    {
        switch (level)
        {
            case sec_level_type::sec128:
                return heongpu_128bit_std_parms(poly_modulus_degree);
            case sec_level_type::sec192:
                return heongpu_192bit_std_parms(poly_modulus_degree);
            case sec_level_type::sec256:
                return heongpu_256bit_std_parms(poly_modulus_degree);
            default:
                return 0;
        }
    }

    namespace
    {
        // Walks the values 1 (mod factor) from start towards end, which is
        // excluded, and returns the next prime not in excluded, or 0.
        class PrimeWalk
        {
          public:
            PrimeWalk(Data64 start, Data64 end, Data64 factor,
                      const std::vector<Data64>& excluded)
                : value_(start), end_(end), factor_(factor), up_(start < end),
                  excluded_(excluded)
            {
            }

            Data64 next()
            {
                while (up_ ? (value_ < end_) : (value_ > end_))
                {
                    Data64 candidate = value_;
                    value_ = up_ ? value_ + factor_ : value_ - factor_;
                    if (is_prime(candidate) &&
                        std::find(excluded_.begin(), excluded_.end(),
                                  candidate) == excluded_.end())
                    {
                        return candidate;
                    }
                }
                return 0;
            }

          private:
            Data64 value_;
            Data64 end_;
            Data64 factor_;
            bool up_;
            const std::vector<Data64>& excluded_;
        };

        void check_prime_request(size_t poly_modulus_degree, int bit_size,
                                 int max_bit_size)
        {
            if (!is_power_of_two(poly_modulus_degree) ||
                (poly_modulus_degree > MAX_POLY_DEGREE) ||
                (poly_modulus_degree < MIN_POLY_DEGREE))
            {
                throw ParameterError(error_message(
                    "Invalid poly modulus degree ", poly_modulus_degree,
                    "! Expected a power of two from ", MIN_POLY_DEGREE, " to ",
                    MAX_POLY_DEGREE, "."));
            }
            // 2N + 1, the smallest candidate, has log2(N) + 2 bits.
            int min_bit_size = calculate_bit_size(poly_modulus_degree) + 1;
            if ((bit_size < min_bit_size) || (bit_size > max_bit_size))
            {
                throw ParameterError(error_message(
                    "Invalid prime bit size ", bit_size, "! Expected ",
                    min_bit_size, " to ", max_bit_size,
                    " for N = ", poly_modulus_degree, "."));
            }
        }

        [[noreturn]] void throw_not_enough_primes(size_t count, int bit_size,
                                                  size_t poly_modulus_degree)
        {
            throw ParameterError(error_message(
                "Not enough primes 1 mod ", 2 * poly_modulus_degree,
                " around 2^", bit_size, " for ", count, " moduli!"));
        }
    } // namespace

    std::vector<Data64>
    generate_primes_below(size_t poly_modulus_degree, int bit_size,
                          size_t count, const std::vector<Data64>& excluded)
    {
        check_prime_request(poly_modulus_degree, bit_size, MAX_MOD_BIT_COUNT);

        Data64 factor = Data64(2) * Data64(poly_modulus_degree);
        Data64 power = Data64(1) << bit_size;
        PrimeWalk below(power - factor + 1, power >> 1, factor, excluded);

        std::vector<Data64> primes;
        while (primes.size() < count)
        {
            Data64 prime = below.next();
            if (prime == 0)
            {
                throw_not_enough_primes(count, bit_size, poly_modulus_degree);
            }
            primes.push_back(prime);
        }
        return primes;
    }

    std::vector<Data64>
    generate_primes_around(size_t poly_modulus_degree, int bit_size,
                           size_t count, const std::vector<Data64>& excluded)
    {
        // The primes above 2^bit_size have one more bit.
        check_prime_request(poly_modulus_degree, bit_size,
                            MAX_MOD_BIT_COUNT - 1);

        Data64 factor = Data64(2) * Data64(poly_modulus_degree);
        Data64 power = Data64(1) << bit_size;
        PrimeWalk below(power - factor + 1, power >> 1, factor, excluded);
        PrimeWalk above(power + 1, power << 1, factor, excluded);

        std::vector<Data64> primes;
        while (primes.size() < count)
        {
            Data64 prime =
                (primes.size() % 2 == 0) ? below.next() : above.next();
            if (prime == 0)
            {
                throw_not_enough_primes(count, bit_size, poly_modulus_degree);
            }
            primes.push_back(prime);
        }
        return primes;
    }

    bool is_primitive_root(Data64 root, size_t degree, Modulus64& modulus)
    {
        // root^(degree/2) = modulus - 1 .
        Data64 degree_over2 = degree >> 1;

        return OPERATOR64::exp(root, degree_over2, modulus) ==
               (modulus.value - 1);
    }

    bool find_primitive_root(size_t degree, Modulus64& modulus,
                             Data64& destination)
    {
        Data64 size_entire_group = modulus.value - 1;

        Data64 size_quotient_group = size_entire_group / degree;

        if (size_entire_group - size_quotient_group * degree != 0)
        {
            return false;
        }

        std::random_device rd;

        int attempt_counter = 0;
        int attempt_counter_max = 100;
        do
        {
            attempt_counter++;

            Data64 random_num =
                (static_cast<Data64>(rd()) << 32) | static_cast<Data64>(rd());
            // destination = OPERATOR64::reduce(random_num, modulus);
            destination = random_num % modulus.value;

            // Raise the random number to power the size of the quotient
            // to get rid of irrelevant part
            destination =
                OPERATOR64::exp(destination, size_quotient_group, modulus);
        } while (!is_primitive_root(destination, degree, modulus) &&
                 (attempt_counter < attempt_counter_max));

        return is_primitive_root(destination, degree, modulus);
    }

    Data64 find_minimal_primitive_root(size_t degree, Modulus64& modulus)
    {
        Data64 root;
        if (!find_primitive_root(degree, modulus, root))
        {
            throw std::logic_error("no sufficient root unity");
        }

        Data64 generator_sq = OPERATOR64::mult(root, root, modulus);

        Data64 current_generator = root;

        for (size_t i = 0; i < degree; i += 2)
        {
            if (current_generator < root)
            {
                root = current_generator;
            }

            current_generator =
                OPERATOR64::mult(current_generator, generator_sq, modulus);
        }

        return root;
    }

    std::vector<Data64>
    generate_primitive_root_of_unity(size_t poly_modulus_degree,
                                     std::vector<Modulus64> primes)
    {
        std::vector<Data64> root_of_unity;

        // 2nth root of unity
        for (int i = 0; i < primes.size(); i++)
        {
            root_of_unity.push_back(find_minimal_primitive_root(
                2 * poly_modulus_degree, primes[i]));
        }

        return root_of_unity;
    }

    std::vector<Root64> generate_ntt_table(std::vector<Data64> psi,
                                           std::vector<Modulus64> primes,
                                           int n_power)
    {
        int n_ = 1 << n_power;
        std::vector<Root64> forward_table; // bit reverse order
        for (int i = 0; i < primes.size(); i++)
        {
            std::vector<Root64> table;
            table.push_back(1);

            for (int j = 1; j < n_; j++)
            {
                Data64 exp =
                    OPERATOR64::mult(table[(j - 1)], psi[i], primes[i]);
                table.push_back(exp);
            }

            for (int j = 0; j < n_; j++) // take bit reverse order
            {
                forward_table.push_back(table[gpuntt::bitreverse(j, n_power)]);
            }
        }

        return forward_table;
    }

    std::vector<Root64> generate_intt_table(std::vector<Data64> psi,
                                            std::vector<Modulus64> primes,
                                            int n_power)
    {
        int n_ = 1 << n_power;
        std::vector<Root64> inverse_table; // bit reverse order
        for (int i = 0; i < primes.size(); i++)
        {
            std::vector<Root64> table;
            table.push_back(1);

            Data64 inv_root = OPERATOR64::modinv(psi[i], primes[i]);
            for (int j = 1; j < n_; j++)
            {
                Data64 exp =
                    OPERATOR64::mult(table[(j - 1)], inv_root, primes[i]);
                table.push_back(exp);
            }

            for (int j = 0; j < n_; j++) // take bit reverse order
            {
                inverse_table.push_back(table[gpuntt::bitreverse(j, n_power)]);
            }
        }

        return inverse_table;
    }

    std::vector<Ninverse64> generate_n_inverse(size_t poly_modulus_degree,
                                               std::vector<Modulus64> primes)
    {
        Data64 n_ = poly_modulus_degree;
        std::vector<Ninverse64> n_inverse;
        for (int i = 0; i < primes.size(); i++)
        {
            n_inverse.push_back(OPERATOR64::modinv(n_, primes[i]));
        }

        return n_inverse;
    }

    __global__ void unsigned_signed_convert(Data64* input, Data64* output,
                                            Modulus64* modulus)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size

        // The values from (t + 1) / 2 up are negative: an odd t gives
        // -(t - 1) / 2 to (t - 1) / 2.
        int64_t threshold = (modulus[0].value + 1) >> 1;
        int64_t input_reg = static_cast<int64_t>(input[idx]);
        input_reg = (input_reg >= threshold)
                        ? input_reg - static_cast<int64_t>(modulus[0].value)
                        : input_reg;

        output[idx] = static_cast<Data64>(input_reg);
    }

    __global__ void fill_device_vector(Data64* vector, Data64 number, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size

        if (idx < size)
        {
            vector[idx] = number;
        }
    }

    // Decomposes rotation indices into baby-step (N2) and giant-step (N1)
    // components using the Baby-Step Giant-Step (BSGS) algorithm. Each rotation
    // index r is split as: r = idx_n1 + idx_n2, where idx_n1 = floor(r/N1)*N1
    // (giant-step) and idx_n2 = r % N1 (baby-step). The result is grouped by
    // giant-step value, and the unique giant-step/baby-step rotations are
    // collected into rot_n1/rot_n2 respectively.
    std::vector<std::vector<int>> bsgs_index(const std::vector<int>& array,
                                             int N1, std::vector<int>& rot_n1,
                                             std::vector<int>& rot_n2)
    {
        std::vector<std::vector<int>> result;
        int n = array.size();

        // Track unique giant-step and baby-step rotation values
        std::map<int, bool> rot_n1_map;
        std::map<int, bool> rot_n2_map;

        // Group rotation indices by their giant-step component
        std::vector<int> temp;
        int prev_idx_n1 = (array[0] / N1) * N1;
        for (const auto& rot : array)
        {
            int idx_n1 = (rot / N1) * N1; // giant-step component
            if (idx_n1 != prev_idx_n1)
            {
                // Start a new group when the giant-step value changes
                result.push_back(temp);
                temp.clear();
                prev_idx_n1 = idx_n1;
            }

            int idx_n2 = rot % N1; // baby-step component

            temp.push_back(idx_n1 + idx_n2);

            rot_n1_map[idx_n1] = true;
            rot_n2_map[idx_n2] = true;
        }

        result.push_back(temp); // push the last group

        // Collect unique giant-step rotations (sorted by map key order)
        for (const auto& [key, _] : rot_n1_map)
        {
            rot_n1.push_back(key);
        }
        // Collect unique baby-step rotations (sorted by map key order)
        for (const auto& [key, _] : rot_n2_map)
        {
            rot_n2.push_back(key);
        }
        return result;
    }

    // Find the N1 and N2 split that gives the closest ratio to the desired
    // ratio
    int find_best_bsgs_split(const std::vector<int>& array, int max_N,
                             float bsgs_ratio)
    {
        int best_N1 = 1;
        float best_diff = std::numeric_limits<float>::max();

        for (int N1 = 1; N1 < max_N; N1 <<= 1)
        {
            std::vector<int> rot_n1;
            std::vector<int> rot_n2;
            bsgs_index(array, N1, rot_n1, rot_n2);

            if (rot_n1.size() <= 1 || rot_n2.size() <= 1)
                continue;

            float current_ratio = float(rot_n2.size()) / float(rot_n1.size());
            float diff = std::abs(current_ratio - bsgs_ratio);

            if (diff < best_diff)
            {
                best_diff = diff;
                best_N1 = N1;
            }
        }

        return best_N1;
    }

    // Find a good N1 and N2 split as close as possible to the desired ratio
    std::vector<std::vector<int>> seperate_func(const std::vector<int>& A,
                                                int slots,
                                                std::vector<int>& rot_n1,
                                                std::vector<int>& rot_n2,
                                                float bsgs_ratio)
    {
        int N1 = find_best_bsgs_split(A, slots, bsgs_ratio);
        auto result = bsgs_index(A, N1, rot_n1, rot_n2);
        return result;
    }

    std::vector<int> unique_sort(const std::vector<int>& input)
    {
        std::set<int> result(input.begin(), input.end());

        return std::vector<int>(result.begin(), result.end());
    }

    BootstrappingConfig::BootstrappingConfig(EncodingMatrixConfig stc_config,
                                             EvalModConfig eval_mod_config,
                                             EncodingMatrixConfig cts_config)
        : CtoS_piece_(cts_config.piece_), StoC_piece_(stc_config.piece_),
          stc_config_(stc_config), eval_mod_config_(eval_mod_config),
          cts_config_(cts_config)
    {
    }

    //////////////////////////
    //////////////////////////

    std::vector<Data64>
    calculate_last_q_modinv(const std::vector<Modulus64>& prime_vector,
                            const int Q_prime_size, const int P_size)
    {
        std::vector<Data64> last_q_modinv;
        for (int i = 0; i < P_size; i++)
        {
            for (int j = 0; j < (Q_prime_size - 1) - i; j++)
            {
                // TODO: Change here for BFV as well!!!
                Data64 temp_ = prime_vector[prime_vector.size() - 1 - i].value %
                               prime_vector[j].value;
                last_q_modinv.push_back(
                    OPERATOR64::modinv(temp_, prime_vector[j]));
            }
        }

        return last_q_modinv;
    }

    std::vector<Data64>
    calculate_half(const std::vector<Modulus64>& prime_vector, const int P_size)
    {
        std::vector<Data64> half;
        for (int i = 0; i < P_size; i++)
        {
            half.push_back(prime_vector[prime_vector.size() - 1 - i].value >>
                           1);
        }

        return half;
    }

    std::vector<Data64>
    calculate_half_mod(const std::vector<Modulus64>& prime_vector,
                       const std::vector<Data64>& half, const int Q_prime_size,
                       const int P_size)
    {
        std::vector<Data64> half_mod;
        for (int i = 0; i < P_size; i++)
        {
            for (int j = 0; j < (Q_prime_size - 1) - i; j++)
            {
                half_mod.push_back(half[i] % prime_vector[j].value);
            }
        }

        return half_mod;
    }

    std::vector<Data64>
    calculate_factor(const std::vector<Modulus64>& prime_vector,
                     const int Q_size, const int P_size)
    {
        std::vector<Data64> factor;
        for (int i = 0; i < P_size; i++)
        {
            for (int j = 0; j < Q_size; j++)
            {
                factor.push_back(
                    prime_vector[prime_vector.size() - 1 - i].value %
                    prime_vector[j].value);
            }
        }

        return factor;
    }

    //////////////////////////
    //////////////////////////

    std::vector<Data64> calculate_Mi(const std::vector<Modulus64>& prime_vector,
                                     const int size)
    {
        std::vector<Data64> result_Mi(size * size, 0ULL);
        for (int i = 0; i < size; i++)
        {
            mpz_t result;
            mpz_init(result);
            mpz_set_ui(result, 1);

            for (int j = 0; j < size; j++)
            {
                if (i != j)
                {
                    mpz_mul_ui(result, result, prime_vector[j].value);
                }
            }

            size_t mul_size;
            uint64_t* ptr = reinterpret_cast<uint64_t*>(mpz_export(
                NULL, &mul_size, -1, sizeof(uint64_t), 0, 0, result));

            for (int j = 0; j < mul_size; j++)
            {
                result_Mi[(size * i) + j] = ptr[j];
            }

            mpz_clear(result);
            free(ptr);
        }

        return result_Mi;
    }

    std::vector<Data64>
    calculate_Mi_inv(const std::vector<Modulus64>& prime_vector, const int size)
    {
        std::vector<Data64> result_Mi_inv;
        for (int i = 0; i < size; i++)
        {
            Data64 temp = 1;
            for (int j = 0; j < size; j++)
            {
                if (i != j)
                {
                    Data64 inner_prime =
                        prime_vector[j].value % prime_vector[i].value;
                    temp = OPERATOR64::mult(temp, inner_prime, prime_vector[i]);
                }
            }
            result_Mi_inv.push_back(OPERATOR64::modinv(temp, prime_vector[i]));
        }

        return result_Mi_inv;
    }

    std::vector<Data64> calculate_M(const std::vector<Modulus64>& prime_vector,
                                    const int size)
    {
        std::vector<Data64> result_M(size, 0ULL);

        mpz_t result;
        mpz_init(result);
        mpz_set_ui(result, 1);

        for (int i = 0; i < size; i++)
        {
            mpz_mul_ui(result, result, prime_vector[i].value);
        }

        size_t mul_size;
        uint64_t* ptr = reinterpret_cast<uint64_t*>(
            mpz_export(NULL, &mul_size, -1, sizeof(uint64_t), 0, 0, result));

        for (int j = 0; j < mul_size; j++)
        {
            result_M[j] = ptr[j];
        }

        mpz_clear(result);
        free(ptr);

        return result_M;
    }

    std::vector<Data64>
    calculate_upper_half_threshold(const std::vector<Modulus64>& prime_vector,
                                   const int size)
    {
        std::vector<Data64> result_upper_half_threshold(size, 0ULL);

        mpz_t result;
        mpz_init(result);
        mpz_set_ui(result, 1);

        for (int i = 0; i < size; i++)
        {
            mpz_mul_ui(result, result, prime_vector[i].value);
        }

        mpz_add_ui(result, result, 1);
        mpz_div_2exp(result, result, 1);

        size_t mul_size;
        uint64_t* ptr = reinterpret_cast<uint64_t*>(
            mpz_export(NULL, &mul_size, -1, sizeof(uint64_t), 0, 0, result));

        for (int j = 0; j < mul_size; j++)
        {
            result_upper_half_threshold[j] = ptr[j];
        }

        mpz_clear(result);
        free(ptr);

        return result_upper_half_threshold;
    }

    void ensure_kernel_stack_size(size_t bytes, bool raise)
    {
        size_t current = 0;
        HEONGPU_CUDA_CHECK(cudaDeviceGetLimit(&current, cudaLimitStackSize));
        if (current >= bytes)
        {
            return;
        }
        if (!raise)
        {
            throw StateError(error_message(
                "The CUDA kernel stack size limit is ", current,
                " bytes, but HEonGPU needs at least ", bytes,
                ". Raise it with cudaDeviceSetLimit(cudaLimitStackSize, ",
                bytes,
                ") or leave MemoryPoolConfig::raise_kernel_stack_size on."));
        }
        HEONGPU_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, bytes));
    }

    //////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////

} // namespace heongpu