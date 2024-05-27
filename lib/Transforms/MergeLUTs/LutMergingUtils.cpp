#include "lib/Transforms/MergeLUTs/LutMergingUtils.h"
#include "llvm/include/llvm/Support/Debug.h"   // from @llvm-project

#include <algorithm>
#include <ostream>
#include <vector>

#define DEBUG_TYPE "merge-luts"

template <class T>
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const std::vector<T>& vec)
{
    os << "[";
    for (auto t : vec) os << t;
    os << "]";
    
    return os;
}

std::vector<bool> intToBits(int value, int width)
{
    std::vector<bool> bits;
    bits.reserve(width);
    for (int i = 0; i < width; i++)
        bits.push_back((value & (1 << (width - i - 1))) != 0);
    return bits;
}

std::vector<bool> intToBitsFlip(int value, int width)
{
    std::vector<bool> bits;
    bits.reserve(width);
    for (int i = 0; i < width; i++)
        bits.push_back((value & (1 << i)) != 0);
    return bits;
}

int bitsToInt(const std::vector<bool>& bits)
{
    int value = 0;
    int power = 0;
    for (const auto& bit : llvm::reverse(bits))
    {
        if (bit) value += (1 << power);
        power++;
    }
    return value;
}

int bitsToIntFlip(const std::vector<bool>& bits)
{
    int value = 0;
    int power = 0;
    for (const auto& bit : bits)
    {
        if (bit) value += (1 << power);
        power++;
    }
    return value;
}

int composeLookupTables(
    const llvm::SmallVector<int>& sourceIdxs, int sourceLut, 
    const llvm::SmallVector<int>& destIdxs, int destLut)
{
    int composedInputCount = std::max(
        *std::max_element(sourceIdxs.begin(), sourceIdxs.end()),
        *std::max_element(destIdxs.begin(), destIdxs.end())) + 1;

    llvm::DenseMap<int, int> sourceLocs;
    llvm::DenseMap<int, int> destLocs;

    std::vector<bool> sourceLutTable = intToBitsFlip(sourceLut, 1 << sourceIdxs.size());
    std::vector<bool> destLutTable = intToBitsFlip(destLut, 1 <<  destIdxs.size());

    LLVM_DEBUG({
        llvm::dbgs() << "Source LUT is " << sourceLutTable << " (" << sourceLut << ")\n";
        llvm::dbgs() << "Dest LUT is " << destLutTable << " (" << destLut << ")\n";
    });

    for (const auto& [loc, index] : llvm::enumerate(sourceIdxs))
        sourceLocs.insert({index, loc});

    for (const auto& [loc, index] : llvm::enumerate(destIdxs))
        destLocs.insert({index, loc});

    std::vector<bool> composedLookupTable;

    for (int possibleInput = 0; possibleInput < (1 << composedInputCount); possibleInput++)
    {
        auto inputs = intToBits(possibleInput, composedInputCount);
        std::vector<bool> sourceInputs;
        for (auto idx : sourceIdxs) sourceInputs.push_back(inputs[idx]);
        // LLVM_DEBUG({llvm::dbgs() << "Source input is " << sourceInputs << " (" << bitsToInt(sourceInputs) << ")\n"; });
        bool sourceOutput = sourceLutTable[bitsToInt(sourceInputs)];

        std::vector<bool> destInputs;

        for (auto idx : destIdxs)
        {
            if (idx == -1) destInputs.push_back(sourceOutput);
            else destInputs.push_back(inputs[idx]);
        }
        
        LLVM_DEBUG({
            llvm::dbgs() << "Possible input: " << possibleInput << " " << intToBits(possibleInput, composedInputCount) << "\n";
            llvm::dbgs() << "\tSource inputs: " << sourceInputs << " " << bitsToInt(sourceInputs) << " -> " << sourceOutput << "\n";
            llvm::dbgs() << "\tDest inputs: " << destInputs << " " << bitsToInt(destInputs) << " -> " << bitsToInt(destInputs) << "\n";
            llvm::dbgs() << "\tOutput: " << destLutTable[bitsToInt(destInputs)] << "\n";
        });
        
        composedLookupTable.push_back(destLutTable[bitsToInt(destInputs)]);
    }

    return bitsToIntFlip(composedLookupTable);
}