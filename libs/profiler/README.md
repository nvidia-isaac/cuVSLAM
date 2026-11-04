NVTX
===================

This library is a wrapping around  [nvtx3](https://github.com/NVIDIA/NVTX) and provide a RAII helpers to trace apps using NVIDIA NSight.

Disclaimer
----------
This lib was tested with hand-installed [NSight Systems 2020.5](https://developer.nvidia.com/rdp/assets/nsight-systems-2020-5h-linux-installer).

Usage
-----
To use the lib you have to include nvtx first.
```
#include "profiler/profiler.h"
```

The lib provides several classes and functions for tracing.

You can trace some event using TraceEventStart/TraceEventEnd functions:

```
// example
auto e_id = TraceEventStart("foo");
foo();
TraceEventEnd(e_id);

// you dont have to end event in corresponding order
auto e1_id = TraceEventStart("foo");
auto e2_id = TraceEventStart("bar");
foo();
TraceEventEnd(e1_id);
bar();
TraceEventEnd(e2_id);
```

There is also a context-wise trace event. The event will automatically end we run out of scope.
```
{
auto ev = TraceEvent("foo");
foo();
}
```

You may also want to mark some specific moment of time.

```
Mark("something_happend");
```

Combine events
--------------

There is also a way to combine events in one distinct set called Domain.

```
auto domain = Domain("my_domain_name");

// create a mark in this domain
DomainMark(domain.get_handle(), "something_happend_inside_my_domain_name");

//also trace an event
auto ev_id DomainTraceEventStart(domain.get_handle(), "track");
tracker->track();
DomainTraceEventEnd(domain.get_handle(), ev_id);
```

Is might seem inconvenient to keep attention usually about domain handle, so I added also a DomainHelper.

```
auto helper = DomainHelper("new_domain");

helper.mark("something_happend_in_new_domain");

{
    auto ev_ = helper.trace_event("scope_event");
    foo();
}

auto ev_id_ = helper.trace_event_start("bar");
bar();
helper.trace_event_end(ev_id);
};

```

Naming threads
--------------

Threads cuVSLAM creates name themselves at the top of their entry function:

```
profiler::NameCurrentThread("cuvslam:sba");
```

This sets the OS thread name (shown by gdb, top, perf) in every build, and the NVTX thread name when
built with `USE_NVTX`. Linux truncates names to 15 characters. A thread must name itself: Nsight Systems
rejects traces where one thread names another. Don't name threads cuVSLAM doesn't own, such as the
caller's thread.

Usage in Nsight Systems.
-----------------------

>- Download and install Nsight Systems
>- ```cd nsight../bin/```
>- ```./nsys-ui```
>- Select the machine you want to trace the app on
>- Fill in the path to executable with arguments
>- Fill in working directory path
>- Fill in enviromental variables
>- Push "Collect NVTX Trace" checkbox
>- Press Start
