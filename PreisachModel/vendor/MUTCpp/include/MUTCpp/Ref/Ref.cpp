// This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <unordered_set>
#include <mutex>
#include "Ref.hpp"

#include <cassert>

namespace mc
{
    static std::unordered_set<void *> s_LiveReferences;
    static std::mutex s_LiveReferenceMutex;

    namespace RefUtils
    {
        void AddToLiveReferences(void *instance)
        {
            std::scoped_lock lock(s_LiveReferenceMutex);
            assert(instance);
            s_LiveReferences.insert(instance);
        }

        void RemoveFromLiveReferences(void *instance)
        {
            std::scoped_lock lock(s_LiveReferenceMutex);
            assert(instance);
            assert(s_LiveReferences.contains(instance));
            s_LiveReferences.erase(instance);
        }

        bool IsLive(void *instance)
        {
            assert(instance);
            return s_LiveReferences.contains(instance);
        }
    }
}
