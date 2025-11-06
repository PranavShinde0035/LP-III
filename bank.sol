// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Bank {
    mapping(address => uint) public balance;

    // Deposit money
    function deposit() public payable {
        balance[msg.sender] += msg.value;
    }

    // Withdraw money
    function withdraw(uint amount) public {
        require(balance[msg.sender] >= amount, "Insufficient balance");
        balance[msg.sender] -= amount;
        payable(msg.sender).transfer(amount);
    }

    // Show balance
    function getBalance() public view returns(uint) {
        return balance[msg.sender];
    }
}
