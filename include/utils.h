//
// Created by root on 7/30/22.
//

#ifndef AIOPERATION_UTILS_H
#define AIOPERATION_UTILS_H

#include <chrono>

// do while的作用之一是防止命名冲突
#define Timing( str ,func ) \
do {                                                                                    \
    auto begin = std::chrono::steady_clock::now();                                      \
    func;                                                                               \
    auto end = std::chrono::steady_clock::now();                                        \
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);  \
    std::cout << "--" << str << " " << duration.count() << " ns " << std::endl;                                                                                   \
} while(0)



#endif //AIOPERATION_UTILS_H
