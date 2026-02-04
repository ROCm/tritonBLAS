<!--
MIT License

Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

[README](../README.md) > **Contributing**

# Contributing to tritonBLAS

Thank you for your interest in contributing to tritonBLAS! This document provides guidelines for contributing to the project.

## Development Setup


1. **Fork the Repository**: Start by forking the tritonBLAS repository on GitHub to your own account.


2. **Clone Your Fork** :
   Clone the repository:
   ```bash
   git clone <YOUR_FORK_URL>
   ```

## Development Workflow

1. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**:
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**:
   ```bash
   # Run code quality checks
   ruff check .
   ruff format .
   
   # Run tests (if available)
   cd <tritonblas_root>/tests
   pytest
   ```

4. **Commit and Push**:
   ```bash
   git add .
   git commit -m "Description of your changes"
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**:
   - Go to the GitHub repository
   - Create a new pull request from your branch
   - Fill in the PR description with details about your changes
   - Feel free to open a draft PR and ask for early feedback while you're still working on your changes

## License

By contributing to tritonBLAS, you agree that your contributions will be licensed under the MIT License. 