// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract StudentData {

    // Structure
    struct Student {
        uint rollNo;
        string name;
        uint marks;
    }

    // Array of students
    Student[] public students;

    // Add student details
    function addStudent(uint _rollNo, string memory _name, uint _marks) public {
        students.push(Student(_rollNo, _name, _marks));
    }

    // Get student count
    function getStudentCount() public view returns(uint) {
        return students.length;
    }

    // Fallback function
    fallback() external payable { }

    receive() external payable { }
}
