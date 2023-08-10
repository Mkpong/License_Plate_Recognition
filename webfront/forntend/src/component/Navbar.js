import React from 'react'
import Container from 'react-bootstrap/Container';
import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';
import { useState , useEffect } from 'react';

function NavBar(){

    return(
      <Container fluid style={{maxWidth: '1300px' , borderRadius: '50px 50px'}} className='mb-2'>
        <Navbar bg="light" expand="lg" style={{borderRadius: '20px 20px'}}>
          <Navbar.Brand href="/" className='mx-2'>Mkpong</Navbar.Brand>
          <Navbar.Toggle aria-controls="navbarScroll" />
          <Navbar.Collapse id="navbarScroll">
            <Nav
              className="me-auto my-2 my-lg-0"
              style={{ maxHeight: '100px' }}
              navbarScroll
            >
              <Nav.Link href="/">Home</Nav.Link>
              <Nav.Link href="/parking/state">ParkingState</Nav.Link>
              <Nav.Link href="/history">History</Nav.Link>
              <Nav.Link href="/season">Season</Nav.Link>
              <Nav.Link href="/preferential">Preferential</Nav.Link>
              <Nav.Link href="/admin">Admin</Nav.Link>
            </Nav>
          </Navbar.Collapse>
      </Navbar>
      </Container>
    );
}

export default NavBar;