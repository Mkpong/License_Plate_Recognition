import React from "react";
import { useMemo, useEffect, useState } from "react";
import SetTable from "./SetTable";
import axios from "axios";
import { Container, Row, Col, Button, Form } from "react-bootstrap";

function History() {
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
            Header: 'Entrance Time'
        },
        {
            accessor: 'departureTime',
            Header: 'Departure Time'
        },
        {
            accessor: 'parkingFee',
            Header: 'Fee'
        }
    ] , [])

      
      const [data , setData] = useState();
      const [total, setTotal] = useState();
      const [totalElements, setTotalElements] = useState(0);
      const [totalPages, setTotalPages] = useState(0);
      const [keyword , setKeyword] = useState("");
      const [page, setPage] = useState(0);


      useEffect(() => {
        searchHistory();
      } , [page])

      const searchHistory = () => {
        axios.post(`/api/history/search?page=${page}&keyword=${keyword}`)
        .then(response => {
            setData(response.data.content);
            setTotalElements(response.data.totalElements);
            setTotalPages(response.data.totalPages);
        })
        .catch(error => console.log(error))
      }

    return (
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
                        <Button size='sm'variant="info" onClick={searchHistory}>Search</Button>
                    </Col>
                </Row>
        </Container>
    );
}

export default History;