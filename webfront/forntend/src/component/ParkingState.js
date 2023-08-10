import React from "react";
import { useMemo, useEffect, useState } from "react";
import SetTable from "./SetTable";
import axios from "axios";
import { Container, Row, Col, Form, Button } from "react-bootstrap";

function ParkingState() {
    const columns = useMemo(() => [
        {
            accessor: 'id',
            Header: 'ID',
        },
        {
            accessor: 'carNumber',
            Header: 'Number',
        },
        {
            accessor: 'entranceTime',
            Header: 'Time'
        },
        {
            accessor: 'st_car.id',
            Header: 'season'
        }
    ] , [])

      
      const [data , setData] = useState();
      const [keyword, setKeyword] = useState("");
      const [totalPages, setTotalPages] = useState(0);
      const [totalElements, setTotalElements] = useState(0);
      const [page , setPage] = useState(0);

      useEffect(() => {
        searchCar();
      } , [page])

      const searchCar = () => {
        axios.post(`/api/parking/state/search?page=${page}&keyword=${keyword}`)
        .then(response => {
            setData(response.data.content);
            setTotalPages(response.data.totalPages);
            setTotalElements(response.data.totalElements);
        })
        .catch(error => console.log(error))
      }

    return (
        <div>
            <Container>
                {totalElements !== 0 ? (
                    <>
                    <Row className='text-end'>
                        <Col><div>Total : {totalElements}</div></Col>
                    </Row>
                    <Row>
                        {data && <SetTable linkdata="ID" data={data} columns={columns} pathdata="/"/>}
                    </Row></>) : (
                    <Row className='text-center'>
                        <Col>주차된 차량이 없습니다.</Col>
                    </Row>
                    )
                }
                {totalPages >= 2 &&
                    <Row>
                        <Col md={{span: 1, offset: 4}} style={{textAlign: 'end'}}>
                            <Button hidden={page === 0}
                                    onClick={() => setPage(page - 1)}
                                    size='sm'
                                    variant='dark'>←</Button>
                        </Col>
                        <Col lg={2} style={{textAlign: 'center'}}>
                            <span>Page {page + 1} of {totalPages}</span>
                        </Col>
                        <Col lg={1}>
                            <Button hidden={page === totalPages - 1}
                                    onClick={() => setPage(page + 1)}
                                    size='sm'
                                    variant='dark'>→</Button>
                        </Col>
                    </Row>
                }
                <Row className='mt-3'>
                    <Col md={{span: 3 , offset:4}}>
                        <Form.Control   size = 'sm'
                                        placeholder="Enter CarNumber"
                                        onChange={(e)=>setKeyword(e.target.value)}
                                        style={{width: '300px'}}
                        ></Form.Control>
                    </Col>
                    <Col lg={1}>
                        <Button size='sm'variant="info" onClick={searchCar}>Search</Button>
                    </Col>
                </Row>
            </Container>
        </div>
    );
}

export default ParkingState;