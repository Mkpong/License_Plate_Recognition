import './App.css';
import {Routes, Route, link} from "react-router-dom"
import Home from './component/Home';
import NavBar from './component/Navbar';
import 'bootstrap/dist/css/bootstrap.min.css';
import { Container } from 'react-bootstrap';
import ParkingState from './component/ParkingState';
import History from './component/History';
import Season from './component/Season';
import Preferential from './component/preferential';


function App() {
  return (
    <Container>
      <NavBar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/parking/state" element={<ParkingState />} />
        <Route path="/history" element={<History />} />
        <Route path="/season" element={<Season />} />
        <Route path ="/preferential" element={<Preferential />} />
      </Routes>
    </Container>
  );
}

export default App;
