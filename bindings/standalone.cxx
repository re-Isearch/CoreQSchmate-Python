#include "Util.hpp"
#include "SentenceStore.hpp"

// The same null bridge supplied by Schmate's standalone src/main.cpp. Indexes
// use their native on-disk sentence store and do not reference any IB objects.
std::unique_ptr<SentenceStore> SentenceStoreFactory::CreateBridgeStore(void *) {
    return nullptr;
}
