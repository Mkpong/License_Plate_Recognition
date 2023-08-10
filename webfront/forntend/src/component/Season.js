import React from "react";
import { useMemo, useEffect, useState } from "react";
import SetTable from "./SetTable";
import axios from "axios";
import { Container, Row, Col, Form, Button } from "react-bootstrap";

function Season() {
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
            accessor: 'validDate',
            Header: 'valid Date'
        },
        // {
        //     Header: 'Delete',
        //     accessor: 'delete'
        // }
    ] , [])

      
      const [data , setData] = useState();
      const [input, setInput] = useState({
        car_number: "",
        auto_pay: false,
        month: ""
      })
      const [writemode, setWritemode] = useState(false);
      const [page, setPage] = useState(0);
      const [totalElements, setTotalElements] = useState(0);
      const [totalPages , setTotalPages] = useState(0);
      const [keyword , setKeyword] = useState("");


      useEffect(() => {
        searchSeasonCar();
      } , [writemode, page])

      const onChange = (e) => {
        const id = e.target.id;
        const value = e.target.value;
        setInput({
            ...input,
            [id]: value,
        })
      }

      const searchSeasonCar = () => {
        axios.post(`/api/season/search?page=${page}&keyword=${keyword}`)
        .then(response => {
            setData(response.data.content);
            setTotalElements(response.data.totalElements);
            setTotalPages(response.data.totalPages);
        })
      }

      const addSeasonCar = () => {
        if(writemode){
            if(input.car_number == "" || input.month==""){
                alert("차량 번호와 정기권 기간을 입력해주세요!");
            }
            else{
                axios.post("/api/season/new", input)
                .then(response => {
                    if(response.data === "Fail"){
                        alert("등록에 실패했습니다(이미 등록된 차량)");
                    }
                    else{
                        alert("차량번호 : " + input.car_number +" " + input.month+  "개월 등록완료")
                    }
                })
                .catch(error => console.log(error))
                setWritemode(false)
            }
        }
        else{
            setWritemode(true)
        }
      }

      const cancel = () => {
        setWritemode(false)
      }

    return (
        <div>
            <Container>
                {writemode ? (
                <Row className='text-end'>
                    <Col className='text-end mt-3 mb-3 d-flex' md={{span: 2, offset: 7}}>
                        <Form.Control
                                    type = 'text'
                                    id = 'car_number'
                                    placeholder = 'Enter CarNumber'
                                    onChange = {onChange}>
                        </Form.Control>
                    </Col>
                    <Col className='text-end mt-3 mb-3 d-flex' lg={1}>
                        <Form.Select id="month" onChange={onChange}>
                            <option value="">----</option>
                            <option value='1'>1</option>
                            <option value='2'>2</option>
                            <option value='3'>3</option>
                            <option value='4'>4</option>
                            <option value='5'>5</option>
                            <option value='6'>6</option>
                            <option value='7'>7</option>
                            <option value='8'>8</option>
                            <option value='9'>9</option>
                            <option value='10'>10</option>
                            <option value='11'>11</option>
                            <option value='12'>12</option>
                        </Form.Select> 
                    </Col>
                    <Col className='text-end mt-3 mb-3 d-flex' lg={1}>
                        <Button onClick={addSeasonCar} variant = "info">Register</Button>
                    </Col>
                    <Col className='text-start mt-3 mb-3 d-flex' lg={1}>
                        <Button onClick={cancel} variant = "info">Cancel</Button>
                    </Col>
                </Row>
                ) : (
                <Row>
                    <Col className='text-end mt-3 mb-3 d-fles' md ={{span: 2 , offset: 10}}>
                        <Button onClick={addSeasonCar} variant = "info">정기권 등록</Button>
                    </Col>
                </Row>
                )}
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
                        <Button size='sm'variant="info" onClick={searchSeasonCar}>Search</Button>
                    </Col>
                </Row>
            </Container>
        </div>
    );
}

export default Season;