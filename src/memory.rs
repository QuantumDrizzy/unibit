// ============================================================================
// FORJA-256 — Memory Subsystem
// ============================================================================
//
// Byte-addressable main memory with load/store at various widths.
// Little-endian byte ordering (like x86, ARM-LE, RISC-V).
//
// v0.1: Flat RAM. v0.2 will add L1/L2 cache hierarchy with hit/miss tracking.
// ============================================================================

/// Main memory — flat byte array, little-endian.
pub struct Memory {
    data: Vec<u8>,
    pub size: usize,
    // Metrics
    pub reads: u64,
    pub writes: u64,
}

impl Memory {
    /// Create memory with given size in bytes (default: 1 MiB)
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0u8; size],
            size,
            reads: 0,
            writes: 0,
        }
    }

    /// Default 1 MiB memory
    pub fn default_size() -> Self {
        Self::new(1024 * 1024)
    }

    // ─── Bounds check ─────────────────────────────────────────────────

    #[inline]
    fn check(&self, addr: u64, width: usize) -> Result<usize, String> {
        let addr = addr as usize;
        if addr + width > self.size {
            Err(format!("memory access out of bounds: addr=0x{:x}, width={}, mem_size=0x{:x}",
                addr, width, self.size))
        } else {
            Ok(addr)
        }
    }

    // ─── Load (read) ─────────────────────────────────────────────────

    pub fn load_u8(&mut self, addr: u64) -> Result<u8, String> {
        let a = self.check(addr, 1)?;
        self.reads += 1;
        Ok(self.data[a])
    }

    pub fn load_i8(&mut self, addr: u64) -> Result<i8, String> {
        Ok(self.load_u8(addr)? as i8)
    }

    pub fn load_u16(&mut self, addr: u64) -> Result<u16, String> {
        let a = self.check(addr, 2)?;
        self.reads += 1;
        Ok(u16::from_le_bytes([self.data[a], self.data[a + 1]]))
    }

    pub fn load_i16(&mut self, addr: u64) -> Result<i16, String> {
        Ok(self.load_u16(addr)? as i16)
    }

    pub fn load_u32(&mut self, addr: u64) -> Result<u32, String> {
        let a = self.check(addr, 4)?;
        self.reads += 1;
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&self.data[a..a + 4]);
        Ok(u32::from_le_bytes(bytes))
    }

    pub fn load_i32(&mut self, addr: u64) -> Result<i32, String> {
        Ok(self.load_u32(addr)? as i32)
    }

    pub fn load_u64(&mut self, addr: u64) -> Result<u64, String> {
        let a = self.check(addr, 8)?;
        self.reads += 1;
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.data[a..a + 8]);
        Ok(u64::from_le_bytes(bytes))
    }

    /// Load full 256-bit quad (4 × u64, little-endian)
    pub fn load_256(&mut self, addr: u64) -> Result<[u64; 4], String> {
        let a = self.check(addr, 32)?;
        self.reads += 1;
        let mut lanes = [0u64; 4];
        for i in 0..4 {
            let off = a + i * 8;
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&self.data[off..off + 8]);
            lanes[i] = u64::from_le_bytes(bytes);
        }
        Ok(lanes)
    }

    // ─── Store (write) ────────────────────────────────────────────────

    pub fn store_u8(&mut self, addr: u64, val: u8) -> Result<(), String> {
        let a = self.check(addr, 1)?;
        self.writes += 1;
        self.data[a] = val;
        Ok(())
    }

    pub fn store_u16(&mut self, addr: u64, val: u16) -> Result<(), String> {
        let a = self.check(addr, 2)?;
        self.writes += 1;
        let bytes = val.to_le_bytes();
        self.data[a..a + 2].copy_from_slice(&bytes);
        Ok(())
    }

    pub fn store_u32(&mut self, addr: u64, val: u32) -> Result<(), String> {
        let a = self.check(addr, 4)?;
        self.writes += 1;
        let bytes = val.to_le_bytes();
        self.data[a..a + 4].copy_from_slice(&bytes);
        Ok(())
    }

    pub fn store_u64(&mut self, addr: u64, val: u64) -> Result<(), String> {
        let a = self.check(addr, 8)?;
        self.writes += 1;
        let bytes = val.to_le_bytes();
        self.data[a..a + 8].copy_from_slice(&bytes);
        Ok(())
    }

    /// Store full 256-bit quad
    pub fn store_256(&mut self, addr: u64, lanes: &[u64; 4]) -> Result<(), String> {
        let a = self.check(addr, 32)?;
        self.writes += 1;
        for i in 0..4 {
            let off = a + i * 8;
            let bytes = lanes[i].to_le_bytes();
            self.data[off..off + 8].copy_from_slice(&bytes);
        }
        Ok(())
    }

    /// Write a slice of bytes at address (for loading program data sections)
    pub fn write_bytes(&mut self, addr: u64, bytes: &[u8]) -> Result<(), String> {
        let a = self.check(addr, bytes.len())?;
        self.data[a..a + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    /// Read a slice of bytes
    pub fn read_bytes(&self, addr: u64, len: usize) -> Result<&[u8], String> {
        let a = addr as usize;
        if a + len > self.size {
            return Err(format!("read_bytes out of bounds: addr=0x{:x}, len={}", a, len));
        }
        Ok(&self.data[a..a + len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_store_u64() {
        let mut mem = Memory::new(1024);
        mem.store_u64(0, 0xDEAD_BEEF_CAFE_BABE).unwrap();
        assert_eq!(mem.load_u64(0).unwrap(), 0xDEAD_BEEF_CAFE_BABE);
    }

    #[test]
    fn test_little_endian() {
        let mut mem = Memory::new(1024);
        mem.store_u32(0, 0x01020304).unwrap();
        assert_eq!(mem.load_u8(0).unwrap(), 0x04); // lowest byte first
        assert_eq!(mem.load_u8(3).unwrap(), 0x01); // highest byte last
    }

    #[test]
    fn test_256bit_load_store() {
        let mut mem = Memory::new(1024);
        let lanes = [0x1111, 0x2222, 0x3333, 0x4444];
        mem.store_256(0, &lanes).unwrap();
        let loaded = mem.load_256(0).unwrap();
        assert_eq!(loaded, lanes);
    }

    #[test]
    fn test_out_of_bounds() {
        let mut mem = Memory::new(64);
        assert!(mem.store_u64(60, 0).is_err()); // 60 + 8 > 64
    }
}
