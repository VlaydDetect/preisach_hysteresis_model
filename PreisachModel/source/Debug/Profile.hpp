// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <tracy/Tracy.hpp>

#define AL_PROFILE_MARK_FRAME			FrameMark;
// NOTE(Peter): Use AL_PROFILE_FUNC ONLY at the top of a function
//				Use AL_PROFILE_SCOPE / AL_PROFILE_SCOPE_DYNAMIC for an inner scope
#define AL_PROFILE_FUNC(...)			ZoneScoped##__VA_OPT__(N(__VA_ARGS__))
#define AL_PROFILE_SCOPE(...)			AL_PROFILE_FUNC(__VA_ARGS__)
#define AL_PROFILE_SCOPE_DYNAMIC(NAME)  ZoneScoped; ZoneName(NAME, strlen(NAME))
#define AL_PROFILE_THREAD(...)          tracy::SetThreadName(__VA_ARGS__)
