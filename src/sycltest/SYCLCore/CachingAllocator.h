#ifndef SYCLCore_CachingAllocator_h
#define SYCLCore_CachingAllocator_h

#include <mutex>
#include <iomanip>
#include <map>
#include <tuple>

#include <CL/sycl.hpp>

// Inspired by cub::CachingDeviceAllocator

// template <typename T>
// T* sycl::malloc_device(size_t count,
//                        const queue& syclQueue,
//                        const property_list &propList = {})

namespace cms::sycltools {

  namespace detail {

    inline constexpr unsigned int power(unsigned int base, unsigned int exponent) {
      unsigned int power = 1;
      while (exponent > 0) {
        if (exponent & 1) {
          power = power * base;
        }
        base = base * base;
        exponent = exponent >> 1;
      }
      return power;
    }

    // format a memory size in B/kB/MB/GB
    inline std::string as_bytes(size_t value) {
      if (value == std::numeric_limits<size_t>::max()) {
        return "unlimited";
      } else if (value >= (1 << 30) and value % (1 << 30) == 0) {
        return std::to_string(value >> 30) + " GB";
      } else if (value >= (1 << 20) and value % (1 << 20) == 0) {
        return std::to_string(value >> 20) + " MB";
      } else if (value >= (1 << 10) and value % (1 << 10) == 0) {
        return std::to_string(value >> 10) + " kB";
      } else {
        return std::to_string(value) + "  B";
      }
    }

  }  // namespace detail
  
  class CachingAllocator {
    public:
      struct CachedBytes {
        size_t free = 0;       // total bytes freed and cached on the device
        size_t live = 0;       // total bytes currently in use on the device
        size_t requested = 0;  // total bytes requested and currently in use on the device
      };
      
      // constructor
      CachingAllocator(
          sycl::queue queue,
          unsigned int binGrowth,
          unsigned int minBin,
          unsigned int maxBin,
          size_t maxCachedBytes,
          double maxCachedFraction,
          const bool debug,
          const bool reuseSameQueueAllocations)
          : queue_(queue),
            binGrowth_(binGrowth),
            minBin_(minBin),
            maxBin_(maxBin),
            maxCachedFraction_(maxCachedFraction),
            minBinBytes_(detail::power(binGrowth, minBin)),
            maxBinBytes_(detail::power(binGrowth, maxBin)),
            maxCachedBytes_(cacheSize(maxCachedBytes, maxCachedFraction)),
            debug_(debug),
            reuseSameQueueAllocations_(reuseSameQueueAllocations) {
        if (debug_) {
          std::ostringstream out;
          out << "SYCL CachingAllocator settings\n"
              << "  bin growth " << binGrowth_ << "\n"
              << "  min bin    " << minBin_ << "\n"
              << "  max bin    " << maxBin_ << "\n"
              << "  resulting bins:\n";
          for (auto bin = minBin_; bin <= maxBin_; ++bin) {
            auto binSize = detail::power(binGrowth_, bin);
            out << "    " << std::right << std::setw(12) << detail::as_bytes(binSize) << '\n';
          }
          out << "  maximum amount of cached memory: " << detail::as_bytes(maxCachedBytes_);
          std::cout << out.str() << std::endl;
        }
      }

      // destructor
      // this should never be called while some memory blocks are still live
      ~CachingAllocator() {
        {
          std::scoped_lock lock(mutex_);
          assert(liveBlocks_.empty());
          assert(cachedBytes_.live == 0);
        }
        freeAllCached();
      }

      // Allocate given number of bytes on the current device associated to given queue
      // NOTE: probably this function could use template: instead void*, we will use T*
      void* allocate(size_t bytes, sycl::queue queue) {
        // create a block descriptor for the requested allocation
        BlockDescriptor block;
        block.queue = queue;
        block.requested = bytes;
        std::tie(block.bin, block.bytes) = findBin(bytes);

        // try to re-use a cached block, or allocate a new buffer
        if (not tryReuseCachedBlock(block)) {
          allocateNewBlock(block);
        }

        return block.d_ptr;
      }

      void free(void* ptr) {
        std::scoped_lock lock(mutex_);

        auto blockIterator = liveBlocks_.find(ptr);
        if (blockIterator == liveBlocks_.end()) {
          std::stringstream ss;
          ss << "Trying to free a non-live block at " << ptr;
          throw std::runtime_error(ss.str());
        }

        // remove the block from the list of live blocks
        BlockDescriptor block = std::move(blockIterator->second);
        liveBlocks_.erase(blockIterator);
        cachedBytes_.live -= block.bytes;
        cachedBytes_.requested -= block.requested;

        bool recache = (cachedBytes_.free + block.bytes <= maxCachedBytes_);
        if (recache) {
          // alpaka::enqueue(*(block.queue), *(block.event));
          cachedBytes_.free += block.bytes;
          cachedBlocks_.insert(std::make_pair(block.bin, block));

          if (debug_) {
            std::ostringstream out;
            out << "\tDevice " << queue_.get_device().get_info<sycl::info::device::name>()
                << " returned " << block.bytes << " bytes at " << ptr << " .\n\t\t " 
                << cachedBlocks_.size() << " available blocks cached ("
                << cachedBytes_.free << " bytes), " << liveBlocks_.size()
                << " live blocks (" << cachedBytes_.live << " bytes) outstanding."
                << std::endl;
            std::cout << out.str() << std::endl;
          }
        } else {
          // the block it is automatically freed because it goes out of scope
          if (debug_) {
            std::ostringstream out;
            out << "\tDevice " << queue_.get_device().get_info<sycl::info::device::name>()
                << " freed " << block.bytes << " bytes at " << ptr << " .\n\t\t "
                << cachedBlocks_.size() << " available blocks cached ("
                << cachedBytes_.free << " bytes), " << liveBlocks_.size()
                << " live blocks (" << cachedBytes_.live << " bytes) outstanding." 
                << std::endl;
            std::cout << out.str() << std::endl;
          }
        }
      }

    private:
      struct BlockDescriptor {
        sycl::queue queue;            // associated queue
        void *d_ptr;                  // poiter to data
        size_t bytes = 0;             // bytes allocated
        size_t bytes_requested = 0;   // bytes requested
        unsigned int bin = 0;         // bin class id, binGrowth^bin is the block size  
      };

      sycl::queue queue_;
      std::mutex mutex_;

      const bool debug_;
      const bool reuseSameQueueAllocations_;

      const unsigned int binGrowth_;    // bin growth factor;
      const unsigned int minBin_;       // the smallest bin is set to binGrowth^minBin bytes;
                                        // smaller allocations are rounded to this value;
      const size_t minBinBytes_;        // bytes of the smallest bin
      const unsigned int maxBin_;       // the largest bin is set to binGrowth^maxBin bytes;
                                        // larger allocations will fail;
      const size_t maxBinBytes_;        // bytes of the bigger bin
      const size_t maxCachedBytes_;     // total storage for the allocator (0 means no limit);
      const double maxCachedFraction_;  // fraction of total device memory taken for the allocator (0 means no limit);
                                        // if both maxCachedBytes and maxCachedFraction are non-zero,
                                        // the smallest resulting value is used;

      using CachedBlocks = std::multimap<unsigned int, BlockDescriptor>;
      using BusyBlocks = std::map<void*, BlockDescriptor>;

      CachedBytes cachedBytes_;
      CachedBlocks cachedBlocks_;  // Set of cached device allocations available for reuse
      BusyBlocks liveBlocks_;      // map of pointers to the live device allocations currently in use

      // return the maximum amount of memory that should be cached on this device
      size_t cacheSize(size_t maxCachedBytes, double maxCachedFraction) const {
        size_t totalMemory = queue_.get_device().get_info<sycl::info::device::global_mem_cache_size>();
        size_t memoryFraction = static_cast<size_t>(maxCachedFraction * totalMemory);
        size_t size = std::numeric_limits<size_t>::max();
        if (maxCachedBytes > 0 and maxCachedBytes < size) {
          size = maxCachedBytes;
        }
        if (memoryFraction > 0 and memoryFraction < size) {
          size = memoryFraction;
        }
        return size;
      }

      // NOTE: check the update of cachedBytes_, probably other field must be update
      void freeAllCached() {
        std::scoped_lock lock(mutex_);

        while (not cachedBlocks_.empty()) {
          auto blockIterator = cachedBlocks_.begin();
          cachedBytes_.free -= blockIterator->second.bytes;

          if (debug_) {
            std::ostringstream out;
            out << "\tDevice " << queue_.get_device().get_info<sycl::info::device::name>()
                << " freed " << blockIterator->second.bytes << " bytes.\n\t\t  " 
                << (cachedBlocks_.size() - 1) << " available blocks cached (" 
                << cachedBytes_.free << " bytes), " << liveBlocks_.size() 
                << " live blocks (" << cachedBytes_.live << " bytes) outstanding."
                << std::endl;
            std::cout << out.str() << std::endl;
          }

          cachedBlocks_.erase(blockIterator);
        }
      }

      // return (bin, bin size)
      std::tuple<unsigned int, size_t> findBin(size_t bytes) const {
        if (bytes < minBinBytes_) {
          return std::make_tuple(minBin_, minBinBytes_);
        }
        if (bytes > maxBinBytes_) {
          throw std::runtime_error("Requested allocation size "
              + std::to_string(bytes) 
              + " bytes is too large for the caching detail with maximum bin " 
              + std::to_string(maxBinBytes_) +
              " bytes. You might want to increase the maximum bin size");
        }

        unsigned int bin = minBin_;
        size_t binBytes = minBinBytes_;
        while (binBytes < bytes) {
          ++bin;
          binBytes *= binGrowth_;
        }

        return std::make_tuple(bin, binBytes);
      }

      bool tryReuseCachedBlock(BlockDescriptor& block) {
        std::scoped_lock lock(mutex_);

        // iterate through the range of cached blocks in the same bin
        const auto [begin, end] = cachedBlocks_.equal_range(block.bin);
        for (auto blockIterator = begin; blockIterator != end; ++blockIterator) {
          // NOTA: in this if condition have to be inserted a check on a completed event:
          // if ((reuseSameQueueAllocations_ and (block.queue == (blockIterator->second.queue))) or event_completed) {
          if (reuseSameQueueAllocations_ and (block.queue == (blockIterator->second.queue))) {
            auto queue = block.queue;
            block = blockIterator->second;
            block.queue = queue;

            // NOTA: event part to be reviewed
            // if the new queue is on different device than the old event, create a new event
            // if (block.device() != alpaka::getDev(*(block.event))) {
            //   block.event = Event{block.device()};
            // }

            // insert the cached block into the live blocks
            liveBlocks_[block.d_ptr] = block;

            // update the accounting information
            cachedBytes_.free -= block.bytes;
            cachedBytes_.live += block.bytes;
            cachedBytes_.requested += block.requested;

            if (debug_) {
              std::ostringstream out;
              out << "\tDevice " << queue_.get_device().get_info<sycl::info::device::name>()
                  << " reused cached block at " << block.d_ptr 
                  << " (" << block.bytes << " bytes)" 
                  << " previously associated with device " 
                  << (blockIterator->second.queue).get_device().get_info<sycl::info::device::name>()
                  << std::endl;
              std::cout << out.str();
            }

            // remove the reused block from the list of cached blocks
            cachedBlocks_.erase(blockIterator);
            return true;
          }
        }

        return false;
      }

      void* allocateBuffer(size_t bytes, sycl::queue const& queue) {
          if(queue.get_device().is_host()){
            return block.d_ptr = malloc_host(block.bytes, block.queue);
          } else {
            return block.d_ptr = malloc_device(block.bytes, block.queue);
          }
      }

      void allocateNewBlock(BlockDescriptor& block) {
        try {
          block.d_ptr = allocateBuffer(block.bytes, block.queue);
        } catch (const sycl::exception &e) {
          // the allocation attempt failed: free all cached blocks on the device and retry
          // NOTE: TODO implement a method that frees only up to block.bytes bytes
          if (debug_) {
            std::ostringstream out;
            out << "\tCaught synchronous SYCL exception:\n" << e.what() << "\n"
                << "\tDevice " << queue_.get_device().get_info<sycl::info::device::name>()
                << " failed to allocate " << block.bytes << " bytes,"
                << " retrying after freeing cached allocations" << std::endl;
            std::cout << out.str() << std::endl;
          }
          freeAllCached();

          // throw an exception if it fails again
          // NOTE: this must be checked
          block.d_ptr = allocateBuffer(block.bytes, block.queue);
        }

        // create a new event associated to the "synchronisation device"
        block.event = Event{block.device()};

        {
          std::scoped_lock lock(mutex_);
          cachedBytes_.live += block.bytes;
          cachedBytes_.requested += block.requested;
          liveBlocks_[block.d_ptr] = block;
        }

        if (debug_) {
          std::ostringstream out;
          out << "\tDevice " << queue_.get_device().get_info<sycl::info::device::name>()
              << " allocated new block at " << block.d_ptr 
              << " (" << block.bytes << " bytes." << std::endl;
          std::cout << out.str() << std::endl;
        }
      }
    
    };

}  // namespace cms::sycltools

#endif //SYCLCore_CachingAllocator_h