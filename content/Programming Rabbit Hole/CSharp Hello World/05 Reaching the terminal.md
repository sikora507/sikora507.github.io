---
title: "Reaching the terminal"
---

Follow the text from the managed output API to the operating system.

## Questions at this layer

- How does this runtime implement console output?
- Where does encoding happen, and how is the output buffered?
- Which operating-system operation writes the bytes, and what arguments does it receive?
- What changes when standard output points to a file instead of a terminal?
- Where does writing bytes end and terminal rendering begin?

[[Programming Rabbit Hole/CSharp Hello World/index|Overview]]
