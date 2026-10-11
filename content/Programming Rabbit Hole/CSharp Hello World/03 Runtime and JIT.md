---
title: "3. Runtime and JIT"
---

Follow how the compiled program starts and reaches native code.

## How can we capture the code generated at runtime?

The JIT translates a method's IL into native machine instructions in memory. We can ask it to print a textual assembly listing as it compiles our entry point.

From the `Hello` project directory, run:

```sh
DOTNET_JitDisasm='Program:<Main>$' \
DOTNET_JitStdOutFile="$PWD/hello-jit.asm" \
DOTNET_TieredCompilation=0 \
dotnet bin/Debug/net10.0/Hello.dll
```

- `DOTNET_JitDisasm` selects our generated entry-point method. The single quotes protect its special characters from the shell.
- `DOTNET_JitStdOutFile` saves the assembly listing to `hello-jit.asm` in the current directory.
- `DOTNET_TieredCompilation=0` disables compilation in multiple tiers for this run.
- `dotnet` starts the runtime and executes our existing Debug build. The program still prints `Hello, World!` to standard output.

These diagnostic options let us inspect the native code without attaching a debugger. See [.NET's JIT diagnostics](https://github.com/dotnet/runtime/blob/main/docs/design/coreclr/jit/viewing-jit-dumps.md) and [runtime compilation settings](https://learn.microsoft.com/en-us/dotnet/core/runtime-config/compilation).

## What did the JIT generate on our machine?

Here is the captured output from our Debug build, running with an ARM64 runtime on Apple Silicon:

```asm
; Assembly listing for method Program:<Main>$(System.String[]) (MinOpts)
; Emitting BLENDED_CODE for generic ARM64 on Apple
; MinOpts code
; debuggable code
; fp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
            stp     fp, lr, [sp, #-0x20]!
            mov     fp, sp
            str     x0, [fp, #0x18]

G_M000_IG02:                ;; offset=0x000C
            movz    x0, #0x8438
            movk    x0, #0xC16 LSL #16
            movk    x0, #1 LSL #32
            ldr     w0, [x0]
            cbz     w0, G_M000_IG04

G_M000_IG03:                ;; offset=0x0020
            bl      CORINFO_HELP_DBG_IS_JUST_MY_CODE

G_M000_IG04:                ;; offset=0x0024
            movz    x0, #0x4650
            movk    x0, #0x947 LSL #16
            movk    x0, #1 LSL #32
            movz    x1, #0x72B8
            movk    x1, #0xC09 LSL #16
            movk    x1, #1 LSL #32
            ldr     x1, [x1]
            blr     x1
            nop
            nop

G_M000_IG05:                ;; offset=0x004C
            ldp     fp, lr, [sp], #0x20
            ret     lr

; Total bytes of code 84
```

This is the native code generated for `<Main>$`, rather than its IL. The header confirms **ARM64 on Apple** and **MinOpts**, meaning minimal optimization. Disabling tiered compilation does not turn a Debug build into an optimized Release build.

The generated method contains **84 bytes of native instructions**, compared with the 12 bytes of IL we inspected earlier. It includes stack-frame setup, debugging support, the output call, and cleanup. It does not include the implementation of `Console.WriteLine` itself or the string object's contents.

The addresses embedded in this listing belong to this particular process run. They can change when we run the same DLL again. A different architecture, runtime version, or build configuration can also produce different instructions.

## Where is the call, and how is the string passed now?

In the IL, `ldstr` put the string reference on the evaluation stack before `call`. In this native listing, the corresponding sequence is in `G_M000_IG04`:

```asm
movz    x0, #0x4650
movk    x0, #0x947 LSL #16
movk    x0, #1 LSL #32
movz    x1, #0x72B8
movk    x1, #0xC09 LSL #16
movk    x1, #1 LSL #32
ldr     x1, [x1]
blr     x1
```

The first three instructions assemble an address in register `x0`, supplying the string reference as the first argument. The next three assemble the address of a call-target slot in `x1`. `ldr x1, [x1]` loads the target from that slot, and **`blr x1` performs the indirect call**, saving a return address in the link register.

This is the native call corresponding to our IL call to `Console.WriteLine(string)`. The target slot can lead through runtime entry stubs; the listing does not name the final target or show its implementation. The operating-system write happens further down the output path.

The other named call, `CORINFO_HELP_DBG_IS_JUST_MY_CODE`, supports debugging. It is separate from the call that prints our message.

## How can we tell where "Hello, World!" comes from?

Look at the first three instructions in `G_M000_IG04`:

```asm
movz    x0, #0x4650
movk    x0, #0x947 LSL #16
movk    x0, #1 LSL #32
```

`movz` initializes the register, and `movk` inserts further 16-bit pieces while preserving the other bits. Together they construct this address:

```text
  0x0000000000004650
| 0x0000000009470000
| 0x0000000100000000
= 0x0000000109474650
```

At the call, `x0` contains the reference passed as the first argument. Comparing this sequence with our IL tells us that it supplies the `"Hello, World!"` string reference. The native listing alone does not show the object's characters. To verify the contents directly, inspect the object at that address in the same running process, or in a memory dump from that run.

## Is the string data in the DLL, and do these instructions read it from there?

The literal's characters originate in **`Hello.dll`'s `#US` user-string heap**. In the IL, this instruction identifies the literal:

```il
ldstr "Hello, World!" /* 70000001 */
```

The token's `0x70` prefix identifies a user string. Its remaining `0x000001` identifies offset 1 within the `#US` heap, rather than an absolute file offset or a runtime memory address.

The runtime resolves the literal to a managed `System.String` object. That object contains runtime type information, a length, and UTF-16 character data. The reference used by native code points to the object, rather than directly to the first character in the DLL.

The `movz` and `movk` instructions above **construct the object reference in a register**. They do not read the characters. The output implementation receives that reference and reads the string's contents.

```text
Hello.dll: #US heap entry
    "Hello, World!"
           ↓ runtime resolves the literal
Runtime memory: System.String object
    length and characters for "Hello, World!"
           ↑ reference supplied in x0
Generated native code
    calls Console.WriteLine(string)
```

## Can the addresses in the generated assembly differ on every run?

Yes. The same DLL can produce native code containing different addresses on different runs. Address-space randomization and runtime allocation affect where code, objects, and call-target slots live.

The instruction pattern can remain the same while its embedded address values change. The address `0x0000000109474650` belongs to our captured run; it is not a permanent location for this literal.

## Is the token 70000001 the same in both IL and native assembly?

No. The token appears in the IL, while our native listing contains a runtime object reference:

| Representation | Value in our example | Meaning |
| --- | --- | --- |
| IL | `70000001` | Identifies the literal in the DLL's user-string heap. |
| Native code | `0x0000000109474650` | References the string object in this process. |

The token stays the same when we run the unchanged DLL. The runtime resolves it to an object, and the JIT generates code that supplies the object's reference. The token itself does not appear in this native listing.

## Did the JIT generate code in memory and also put the string somewhere?

The **JIT and runtime work together**. During compilation, the JIT requests a reference for the literal, and the runtime supplies the string object, creating or reusing it as needed. The JIT emits native instructions that use that reference.

Modern .NET can allocate string literals in a nonmoving, frozen heap. This allows the JIT to embed their object references directly in generated code, as in our listing. Ordinary managed objects can move during garbage collection, so this direct-address pattern should not be assumed for every object. See the runtime's [Non-GC Heap design](https://github.com/dotnet/runtime/blob/main/docs/design/features/NonGC-Heap.md).

The JIT does not rewrite `Hello.dll` into a native executable. The original IL remains in the DLL, while the generated native code and runtime objects live in process memory.

## Can we inspect the whole module, with code and data in one place?

We can capture a **full process memory dump**, but a running .NET module is not one contiguous block containing everything it uses. Its loaded image, generated code, runtime objects, and dependencies occupy different memory regions:

```text
Process memory
├── Loaded Hello.dll image
│   ├── IL
│   ├── Metadata
│   └── #US heap containing "Hello, World!"
├── JIT-generated native code
│   └── <Main>$ instructions
├── Runtime objects
│   └── System.String containing "Hello, World!"
├── Loaded .NET libraries and their code
└── Thread stacks and runtime bookkeeping
```

References connect these regions. Some objects and runtime resources can also be shared across modules, so there is no simple boundary around everything that belongs exclusively to `Hello.dll`.

A full dump puts the process's memory snapshot into **one file**. Diagnostic tools can use it to inspect modules, generated methods, objects, and their addresses. It is a snapshot of one moment, rather than an executable or a single assembly-language listing. Microsoft's [`dotnet-dump` documentation](https://learn.microsoft.com/en-us/dotnet/core/diagnostics/dotnet-dump) describes collecting and analyzing dumps on macOS, Linux, and Windows.

For this example, the workflow is:

1. Add `Console.ReadLine();` after `Console.WriteLine("Hello, World!");` so the process stays alive while we inspect it.
2. Rebuild and run with the JIT diagnostic settings above.
3. Collect a **Full** process dump while the program waits for input.
4. Inspect the generated entry-point method and the string object referenced by its native code.

Rebuilding with `Console.ReadLine` changes the method, so its new listing will differ from the one above. Keep the **JIT listing and dump from the same run** so their addresses correspond. Together they let us connect the DLL's literal, its runtime string object, and the native instructions in one captured process.

[[Programming Rabbit Hole/CSharp Hello World/index|Overview]] · Next: [[Programming Rabbit Hole/CSharp Hello World/04 Native instructions|Native instructions]]
