import React, { useEffect, useState } from "react";
import { Container, Row, Col, Form, Button } from "react-bootstrap";
import { useMemo } from "react";
import axios from "axios";
import SetTable from "./SetTable";

function Preferential() {

    const columns = useMemo(() => [
        {
            accessor: 'id',
            Header: 'ID',
        },
        {
            accessor: 'carNumber',
            Header: 'Number',
        },
    ])

          const [page, setPage] = useState(0);
      const [totalPages, setTotalPages] = useState(0);
      const [totalElements, setTotalElements] = useState(0);
      const [data , setData] = useState();
      const [keyword, setKeyword] = useState("");
      const [writemode, setWritemode] = useState(false);
      const [carNumber , setCarNumber] = useState("");

    useEffect(() => {
        searchCar();
      } , [page, writemode])

      const onChange = (e) => {
        setCarNumber(e.target.value);
      }

      const cancel = () => {
        setWritemode(false);
      }

      const addPreferentialCar = () => {
        if(writemode){
            if(carNumber === ""){
                alert("차량 번호를 입력하세요");
            }
            else{
                axios.post(`/api/preferential/new?carnumber=${carNumber}`)
                .then(response => {
                    if(response.data === "Fail"){
                        alert("차량 등록에 실패하셨습니다(이미 존재하는 차량번호)");
                    }
                    else{
                        alert(carNumber + "등록 완료")
                        setWritemode(false);
                    }
                })
            }
            
        }
        else{
            setWritemode(true)
        }
      }

      const searchCar = () => {
        axios.get(`/api/preferential/search?page=${page}&keyword=${keyword}`)
        .then(response => {
            setData(response.data.content);
            setTotalElements(response.data.totalElements);
            setTotalPages(response.data.totalPages);
        })
        .catch(error => console.log(error));
      }

    return (
        <Container>
                {writemode ? (
                <Row className='text-end'>
                    <Col className='text-end mt-3 mb-3 d-flex' md={{span: 2, offset: 8}}>
                        <Form.Control
                                    type = 'text'
                                    id = 'car_number'
                                    placeholder = 'Enter CarNumber'
                                    onChange = {onChange}>
                        </Form.Control>
                    </Col>
                    <Col className='text-end mt-3 mb-3 d-flex' lg={1}>
                        <Button onClick={addPreferentialCar} variant = "info">Register</Button>
                    </Col>
                    <Col className='text-start mt-3 mb-3 d-flex' lg={1}>
                        <Button onClick={cancel} variant = "info">Cancel</Button>
                    </Col>
                </Row>
                ) : (
                <Row>
                    <Col className='text-end mt-3 mb-3 d-fles' md ={{span: 2 , offset: 10}}>
                        <Button onClick={addPreferentialCar} variant = "info">우대 등록</Button>
                    </Col>
                </Row>
                )}
            {totalElements == 0 ? (
                <Row>차량 내역을 찾을 수 없습니다</Row>
            ) : (
                <>
                <Row className="text-end">
                    <Col><div>Total : {totalElements}</div></Col>
                </Row>
                <Row>
                {data && <SetTable linkdata="ID" data={data} columns={columns} pathdata="/"/>}
                </Row>
                </>
            )}
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
    );
}

export default Preferential;