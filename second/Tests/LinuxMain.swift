import XCTest

import secondTests

var tests = [XCTestCaseEntry]()
tests += secondTests.allTests()
XCTMain(tests)
